"""Native clause-constrained rational-flat operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.rational_flats._kernel import (
    classify_clause_constrained_rational_flats_kernel,
)
from jacobian.math.combinatorics.matroids.rational_flats._models import (
    ClauseConstrainedRationalFlatClassification,
    ClauseConstrainedRationalFlatProblem,
)


def classify_clause_constrained_rational_flats(
    problem: ClauseConstrainedRationalFlatProblem,
) -> ClauseConstrainedRationalFlatClassification:
    """Classify every admitted clause-constrained rational flat modulo symmetry."""

    try:
        return classify_clause_constrained_rational_flats_kernel(problem)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("problem",),
            code=exc.type,
            message=exc.message(),
        ) from exc


__all__ = ["classify_clause_constrained_rational_flats"]
