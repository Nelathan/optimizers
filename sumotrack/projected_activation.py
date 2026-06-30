from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function


class ProjectedActivationGradientSink:
    """Collect projected weight gradients emitted by custom backward paths.

    This is intentionally tiny and explicit. It is a prototype-side channel for
    tensors that cannot legally live in ``Parameter.grad`` because their shape is
    projected rather than parameter-shaped.
    """

    def __init__(self) -> None:
        self.projected_grads: dict[Any, Tensor] = {}

    def add_(self, key: Any, projected_grad: Tensor) -> None:
        existing = self.projected_grads.get(key)
        if existing is None:
            self.projected_grads[key] = projected_grad.detach()
        else:
            existing.add_(projected_grad.detach())

    def clear(self) -> None:
        self.projected_grads.clear()


class _ProjectedActivationLinear(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        basis: Tensor,
        sink: ProjectedActivationGradientSink | MutableMapping[Any, Tensor],
        key: Any,
    ) -> Tensor:
        if input.shape[-1] != weight.shape[1]:
            raise ValueError(f"input feature dim {input.shape[-1]} does not match weight shape {tuple(weight.shape)}")
        if basis.ndim != 2 or basis.shape[1] != weight.shape[1]:
            raise ValueError(f"activation basis must have shape [rank, in_features], got {tuple(basis.shape)} for weight {tuple(weight.shape)}")

        projected_input = input.reshape(-1, input.shape[-1]) @ basis.mT
        ctx.save_for_backward(projected_input, weight)
        ctx.input_shape = tuple(input.shape)
        ctx.has_bias = bias is not None
        ctx.sink = sink
        ctx.key = key

        output = input @ weight.mT
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, Tensor | None, None, None, None]:
        projected_input, weight = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        projected_weight_grad = grad_output_flat.mT @ projected_input

        sink = ctx.sink
        _sink_add(sink, ctx.key, projected_weight_grad)

        grad_input = grad_output_flat @ weight
        grad_input = grad_input.reshape(ctx.input_shape)
        grad_bias = grad_output_flat.sum(dim=0) if ctx.has_bias else None
        return grad_input, None, grad_bias, None, None, None


def projected_activation_linear(
    input: Tensor,
    weight: Tensor,
    basis: Tensor,
    sink: ProjectedActivationGradientSink | MutableMapping[Any, Tensor],
    key: Any,
    bias: Tensor | None = None,
) -> Tensor:
    """Full-rank linear forward with activation-facing projected weight grad.

    ``weight`` follows PyTorch storage coordinates ``[out_features, in_features]``.
    ``basis`` is always activation-facing/storage-right with shape
    ``[rank, in_features]``. During backward this emits
    ``grad_output.T @ (input @ basis.T)``, exactly equivalent to
    ``full_weight_grad @ basis.T`` without returning a full weight gradient.
    """

    return _ProjectedActivationLinear.apply(input, weight, bias, basis, sink, key)


class _ProjectedActivationGatedMlp(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        input_basis: Tensor,
        hidden_basis: Tensor,
        sink: ProjectedActivationGradientSink | MutableMapping[Any, Tensor],
        gate_key: Any,
        up_key: Any,
        down_key: Any,
    ) -> Tensor:
        if input.shape[-1] != gate_weight.shape[1] or input.shape[-1] != up_weight.shape[1]:
            raise ValueError("gate/up weights must consume the input feature dimension")
        if gate_weight.shape != up_weight.shape:
            raise ValueError(f"gate and up weights must have the same shape, got {tuple(gate_weight.shape)} and {tuple(up_weight.shape)}")
        if down_weight.shape[1] != gate_weight.shape[0]:
            raise ValueError("down weight must consume the gated intermediate dimension")
        if input_basis.ndim != 2 or input_basis.shape[1] != input.shape[-1]:
            raise ValueError(f"input basis must have shape [rank, hidden], got {tuple(input_basis.shape)}")
        if hidden_basis.ndim != 2 or hidden_basis.shape[1] != gate_weight.shape[0]:
            raise ValueError(f"hidden basis must have shape [rank, intermediate], got {tuple(hidden_basis.shape)}")

        flat_input = input.reshape(-1, input.shape[-1])
        gate_pre = input @ gate_weight.mT
        up = input @ up_weight.mT
        hidden = torch.nn.functional.silu(gate_pre) * up
        output = hidden @ down_weight.mT

        projected_input = flat_input @ input_basis.mT
        projected_hidden = hidden.reshape(-1, hidden.shape[-1]) @ hidden_basis.mT
        ctx.save_for_backward(projected_input, projected_hidden, gate_pre, up, gate_weight, up_weight, down_weight)
        ctx.input_shape = tuple(input.shape)
        ctx.sink = sink
        ctx.keys = (gate_key, up_key, down_key)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, None, None, None, None, None, None, None, None]:
        projected_input, projected_hidden, gate_pre, up, gate_weight, up_weight, down_weight = ctx.saved_tensors
        gate_key, up_key, down_key = ctx.keys
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])

        down_projected_grad = grad_output_flat.mT @ projected_hidden
        grad_hidden = (grad_output_flat @ down_weight).reshape_as(up)

        silu_gate = torch.nn.functional.silu(gate_pre)
        grad_up = grad_hidden * silu_gate
        grad_up_flat = grad_up.reshape(-1, grad_up.shape[-1])
        up_projected_grad = grad_up_flat.mT @ projected_input
        grad_input_flat = grad_up_flat @ up_weight
        del silu_gate, grad_up, grad_up_flat

        sigmoid_gate = torch.sigmoid(gate_pre)
        silu_grad = sigmoid_gate * (1.0 + gate_pre * (1.0 - sigmoid_gate))
        grad_hidden.mul_(up).mul_(silu_grad)
        grad_gate_flat = grad_hidden.reshape(-1, grad_hidden.shape[-1])
        gate_projected_grad = grad_gate_flat.mT @ projected_input
        grad_input_flat.add_(grad_gate_flat @ gate_weight)
        del sigmoid_gate, silu_grad, grad_gate_flat

        _sink_add(ctx.sink, gate_key, gate_projected_grad)
        _sink_add(ctx.sink, up_key, up_projected_grad)
        _sink_add(ctx.sink, down_key, down_projected_grad)

        grad_input = grad_input_flat.reshape(ctx.input_shape)
        return grad_input, None, None, None, None, None, None, None, None, None


def projected_activation_gated_mlp(
    input: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    input_basis: Tensor,
    hidden_basis: Tensor,
    sink: ProjectedActivationGradientSink | MutableMapping[Any, Tensor],
    gate_key: Any,
    up_key: Any,
    down_key: Any,
) -> Tensor:
    """SwiGLU-style MLP with activation-facing projected weight grads.

    The exact backward here saves full ``gate_pre`` and ``up`` because those are
    required to differentiate the gate without recomputation. It deliberately
    does not save full input or full hidden for weight-gradient formation.
    """

    return _ProjectedActivationGatedMlp.apply(
        input,
        gate_weight,
        up_weight,
        down_weight,
        input_basis,
        hidden_basis,
        sink,
        gate_key,
        up_key,
        down_key,
    )


def _sink_add(sink: ProjectedActivationGradientSink | MutableMapping[Any, Tensor], key: Any, projected_grad: Tensor) -> None:
    if isinstance(sink, ProjectedActivationGradientSink):
        sink.add_(key, projected_grad)
        return
    existing = sink.get(key)
    if existing is None:
        sink[key] = projected_grad.detach()
    else:
        existing.add_(projected_grad.detach())
