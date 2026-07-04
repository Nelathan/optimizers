from __future__ import annotations

from collections.abc import Callable, MutableMapping
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


class OptimizerProjectedGradientSink:
    """Send projected gradients directly into an optimizer queue."""

    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def add_(self, key: Any, projected_grad: Tensor) -> None:
        self.optimizer.queue_projected_grad(key, projected_grad)


ProjectedGradientSink = ProjectedActivationGradientSink | OptimizerProjectedGradientSink | MutableMapping[Any, Tensor]
RightLinearBackwardKernel = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
LeftLinearBackwardKernel = Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
GatedMlpBackwardKernel = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]


_compiled_right_linear_backward: RightLinearBackwardKernel | None = None
_compiled_left_linear_backward: LeftLinearBackwardKernel | None = None
_compiled_gated_mlp_backward: GatedMlpBackwardKernel | None = None


def set_projected_activation_compile(enabled: bool) -> None:
    """Compile pure projected-activation tensor kernels when the harness compiles.

    The custom autograd Functions still own the Python side-channel that queues
    projected gradients into SumoTrack. Only the tensor math inside backward is
    compiled here, keeping optimizer state/bookkeeping out of Dynamo's graph.
    """

    global _compiled_right_linear_backward, _compiled_left_linear_backward, _compiled_gated_mlp_backward
    if enabled:
        if _compiled_right_linear_backward is None:
            _compiled_right_linear_backward = torch.compile(_projected_activation_right_linear_backward_tensors)
        if _compiled_left_linear_backward is None:
            _compiled_left_linear_backward = torch.compile(_projected_activation_left_linear_backward_tensors)
        if _compiled_gated_mlp_backward is None:
            _compiled_gated_mlp_backward = torch.compile(_projected_activation_gated_mlp_backward_tensors)
    else:
        _compiled_right_linear_backward = None
        _compiled_left_linear_backward = None
        _compiled_gated_mlp_backward = None


def _right_linear_backward_kernel() -> RightLinearBackwardKernel:
    return _compiled_right_linear_backward or _projected_activation_right_linear_backward_tensors


def _left_linear_backward_kernel() -> LeftLinearBackwardKernel:
    return _compiled_left_linear_backward or _projected_activation_left_linear_backward_tensors


def _gated_mlp_backward_kernel() -> GatedMlpBackwardKernel:
    return _compiled_gated_mlp_backward or _projected_activation_gated_mlp_backward_tensors


def _projected_activation_right_linear_backward_tensors(
    grad_output_flat: Tensor,
    projected_input: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    projected_weight_grad = grad_output_flat.mT @ projected_input
    grad_input_flat = grad_output_flat @ weight
    return projected_weight_grad, grad_input_flat


def _projected_activation_left_linear_backward_tensors(
    grad_output_flat: Tensor,
    saved_input_flat: Tensor,
    basis: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    projected_grad_output = grad_output_flat @ basis
    projected_weight_grad = projected_grad_output.mT @ saved_input_flat
    grad_input_flat = grad_output_flat @ weight
    return projected_weight_grad, grad_input_flat


def _projected_activation_gated_mlp_backward_tensors(
    grad_output: Tensor,
    projected_gate_input: Tensor,
    projected_up_input: Tensor,
    projected_hidden: Tensor,
    saved_input: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
    down_projected_grad = grad_output_flat.mT @ projected_hidden
    gate_pre = saved_input @ gate_weight.mT
    grad_hidden = (grad_output_flat @ down_weight).reshape(gate_pre.shape)

    silu_gate = torch.nn.functional.silu(gate_pre)
    grad_up = grad_hidden * silu_gate
    grad_up_flat = grad_up.reshape(-1, grad_up.shape[-1])
    up_projected_grad = grad_up_flat.mT @ projected_up_input
    grad_input_flat = grad_up_flat @ up_weight
    del silu_gate, grad_up, grad_up_flat

    sigmoid_gate = torch.sigmoid(gate_pre)
    silu_grad = sigmoid_gate * (1.0 + gate_pre * (1.0 - sigmoid_gate))
    grad_hidden.mul_(silu_grad)
    del sigmoid_gate, silu_grad
    up = saved_input @ up_weight.mT
    grad_hidden.mul_(up)
    del up
    grad_gate_flat = grad_hidden.reshape(-1, grad_hidden.shape[-1])
    gate_projected_grad = grad_gate_flat.mT @ projected_gate_input
    grad_input_flat.add_(grad_gate_flat @ gate_weight)
    del gate_pre, grad_gate_flat

    return grad_input_flat.reshape(saved_input.shape), gate_projected_grad, up_projected_grad, down_projected_grad


class _ProjectedLinear(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        basis: Tensor,
        sink: ProjectedGradientSink,
        key: Any,
        side: str,
    ) -> Tensor:
        if input.shape[-1] != weight.shape[1]:
            raise ValueError(f"input feature dim {input.shape[-1]} does not match weight shape {tuple(weight.shape)}")
        if side == "right":
            if basis.ndim != 2 or basis.shape[1] != weight.shape[1]:
                raise ValueError(f"right/activation basis must have shape [rank, in_features], got {tuple(basis.shape)} for weight {tuple(weight.shape)}")
            projected_input = input.reshape(-1, input.shape[-1]) @ basis.mT
            ctx.save_for_backward(projected_input, weight)
        elif side == "left":
            if basis.ndim != 2 or basis.shape[0] != weight.shape[0]:
                raise ValueError(f"left/loss-gradient basis must have shape [out_features, rank], got {tuple(basis.shape)} for weight {tuple(weight.shape)}")
            # Left-side projection cannot shrink the saved activation: the
            # projectable tensor is grad_output, which only exists in backward.
            # This path preserves residual-facing geometry and avoids returning
            # a full weight.grad, but it has different memory economics from the
            # right/activation-facing path.
            ctx.save_for_backward(input.reshape(-1, input.shape[-1]), basis, weight)
        else:
            raise ValueError(f"projected linear side must be 'right' or 'left', got {side!r}")

        ctx.input_shape = tuple(input.shape)
        ctx.has_bias = bias is not None
        ctx.sink = sink
        ctx.key = key
        ctx.side = side

        output = input @ weight.mT
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, Tensor | None, None, None, None, None]:
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        if ctx.side == "right":
            projected_input, weight = ctx.saved_tensors
            projected_weight_grad, grad_input_flat = _right_linear_backward_kernel()(grad_output_flat, projected_input, weight)
        else:
            saved_input_flat, basis, weight = ctx.saved_tensors
            projected_weight_grad, grad_input_flat = _left_linear_backward_kernel()(grad_output_flat, saved_input_flat, basis, weight)

        sink = ctx.sink
        _sink_add(sink, ctx.key, projected_weight_grad)

        grad_input = grad_input_flat.reshape(ctx.input_shape)
        grad_bias = grad_output_flat.sum(dim=0) if ctx.has_bias else None
        return grad_input, None, grad_bias, None, None, None, None


def projected_activation_linear(
    input: Tensor,
    weight: Tensor,
    basis: Tensor,
    sink: ProjectedGradientSink,
    key: Any,
    bias: Tensor | None = None,
    side: str = "right",
) -> Tensor:
    """Full-rank linear forward with side-aware projected weight grad.

    ``weight`` follows PyTorch storage coordinates ``[out_features, in_features]``.
    For ``side="right"``, ``basis`` is activation-facing/storage-right with
    shape ``[rank, in_features]`` and backward emits
    ``grad_output.T @ (input @ basis.T)``, equivalent to
    ``full_weight_grad @ basis.T``. For ``side="left"``, ``basis`` is
    loss-gradient/storage-left with shape ``[out_features, rank]`` and backward
    emits ``(grad_output @ basis).T @ input``, equivalent to
    ``basis.T @ full_weight_grad``. In both cases the full-rank forward and exact
    input/bias gradients are preserved, while no full weight gradient is
    returned through autograd.
    """

    return _ProjectedLinear.apply(input, weight, bias, basis, sink, key, side)


class _ProjectedActivationGatedMlp(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        gate_input_basis: Tensor,
        up_input_basis: Tensor,
        hidden_basis: Tensor,
        sink: ProjectedGradientSink,
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
        if gate_input_basis.ndim != 2 or gate_input_basis.shape[1] != input.shape[-1]:
            raise ValueError(f"gate input basis must have shape [rank, hidden], got {tuple(gate_input_basis.shape)}")
        if up_input_basis.ndim != 2 or up_input_basis.shape[1] != input.shape[-1]:
            raise ValueError(f"up input basis must have shape [rank, hidden], got {tuple(up_input_basis.shape)}")
        if hidden_basis.ndim != 2 or hidden_basis.shape[1] != gate_weight.shape[0]:
            raise ValueError(f"hidden basis must have shape [rank, intermediate], got {tuple(hidden_basis.shape)}")

        flat_input = input.reshape(-1, input.shape[-1])
        gate_pre = input @ gate_weight.mT
        up = input @ up_weight.mT
        hidden = torch.nn.functional.silu(gate_pre) * up
        output = hidden @ down_weight.mT

        projected_gate_input = flat_input @ gate_input_basis.mT
        projected_up_input = flat_input @ up_input_basis.mT
        projected_hidden = hidden.reshape(-1, hidden.shape[-1]) @ hidden_basis.mT
        ctx.save_for_backward(projected_gate_input, projected_up_input, projected_hidden, input, gate_weight, up_weight, down_weight)
        ctx.input_shape = tuple(input.shape)
        ctx.sink = sink
        ctx.keys = (gate_key, up_key, down_key)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, None, None, None, None, None, None, None, None, None]:
        projected_gate_input, projected_up_input, projected_hidden, input, gate_weight, up_weight, down_weight = ctx.saved_tensors
        gate_key, up_key, down_key = ctx.keys
        grad_input, gate_projected_grad, up_projected_grad, down_projected_grad = _gated_mlp_backward_kernel()(
            grad_output,
            projected_gate_input,
            projected_up_input,
            projected_hidden,
            input,
            gate_weight,
            up_weight,
            down_weight,
        )

        _sink_add(ctx.sink, gate_key, gate_projected_grad)
        _sink_add(ctx.sink, up_key, up_projected_grad)
        _sink_add(ctx.sink, down_key, down_projected_grad)

        return grad_input, None, None, None, None, None, None, None, None, None, None


def projected_activation_gated_mlp(
    input: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    gate_input_basis: Tensor,
    up_input_basis: Tensor,
    hidden_basis: Tensor,
    sink: ProjectedGradientSink,
    gate_key: Any,
    up_key: Any,
    down_key: Any,
) -> Tensor:
    """SwiGLU-style MLP with activation-facing projected weight grads.

    The backward saves projected activations for weight-gradient formation and
    recomputes the gated intermediates needed for the SwiGLU derivative. This
    avoids storing full ``gate_pre`` and ``up`` from the forward pass.
    """

    return _ProjectedActivationGatedMlp.apply(
        input,
        gate_weight,
        up_weight,
        down_weight,
        gate_input_basis,
        up_input_basis,
        hidden_basis,
        sink,
        gate_key,
        up_key,
        down_key,
    )


def _sink_add(sink: ProjectedGradientSink, key: Any, projected_grad: Tensor) -> None:
    if isinstance(sink, (ProjectedActivationGradientSink, OptimizerProjectedGradientSink)):
        sink.add_(key, projected_grad)
        return
    existing = sink.get(key)
    if existing is None:
        sink[key] = projected_grad.detach()
    else:
        existing.add_(projected_grad.detach())
