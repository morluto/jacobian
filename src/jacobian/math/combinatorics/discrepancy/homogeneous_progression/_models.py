"""Typed contracts for the homogeneous progression set system constructor."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.combinatorics.discrepancy._models import (
    MAX_GROUND_SET,
    FiniteSetSystem,
)


class HomogeneousProgressionRequest(StrictModel):
    """Request to construct the homogeneous progression set system."""

    n: int = Field(ge=0, le=MAX_GROUND_SET)


class HomogeneousProgressionResult(StrictModel):
    """The canonical homogeneous progression set system."""

    n: int
    set_system: FiniteSetSystem


__all__ = [
    "HomogeneousProgressionRequest",
    "HomogeneousProgressionResult",
]
