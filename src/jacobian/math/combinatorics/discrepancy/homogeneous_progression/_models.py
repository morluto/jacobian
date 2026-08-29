"""Typed contracts for the homogeneous progression set system constructor."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.discrepancy._models import FiniteSetSystem


class HomogeneousProgressionRequest(StrictModel):
    """Request to construct the homogeneous progression set system."""

    n: int


class HomogeneousProgressionResult(StrictModel):
    """The canonical homogeneous progression set system."""

    n: int
    set_system: FiniteSetSystem


__all__ = [
    "HomogeneousProgressionRequest",
    "HomogeneousProgressionResult",
]
