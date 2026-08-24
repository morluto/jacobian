"""Domain adapter for finite delta-matroid recognition."""

from __future__ import annotations

from jacobian.math.delta_matroids._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
)
from jacobian.math.delta_matroids.operations import from_feasible_sets

__all__ = ["compute_from_feasible_sets"]


def compute_from_feasible_sets(
    request: DeltaMatroidFromFeasibleSetsRequest,
) -> DeltaMatroidRecognitionResult:
    """Recognize a complete feasible family as a finite delta-matroid."""

    return from_feasible_sets(request.system)
