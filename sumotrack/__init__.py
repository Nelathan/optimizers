"""SUMOTrack optimizer experiments.

The first public optimizer is expected to be ``SubspaceMuon``.  Stage one only
exports the projector machinery; the optimizer itself comes after the math is
nailed down.
"""

from .projector import ProjectionSide, SubspaceProjector
from .optimizer import SubspaceMuon
from .rotation import RoundRobinRefreshScheduler

__all__ = ["ProjectionSide", "RoundRobinRefreshScheduler", "SubspaceMuon", "SubspaceProjector"]
