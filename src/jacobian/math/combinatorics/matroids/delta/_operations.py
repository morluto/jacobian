"""Domain adapter for finite delta-matroid recognition."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidFromFeasibleSetsRequest,
    DeltaMatroidRecognitionResult,
)
from jacobian.math.combinatorics.matroids.delta.operations import from_feasible_sets
from jacobian.math.combinatorics.matroids.delta.values import DeltaMatroidAdmissionError

__all__ = ["compute_from_feasible_sets"]


def compute_from_feasible_sets(
    request: DeltaMatroidFromFeasibleSetsRequest,
) -> DeltaMatroidRecognitionResult:
    """Recognize a complete feasible family as a finite delta-matroid."""

    try:
        return from_feasible_sets(request.system)
    except DeltaMatroidAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("system",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
