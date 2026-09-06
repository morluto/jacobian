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
    RationalFlatOrbitRepresentative,
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


def verify_rational_flat_classification(
    claim: ClauseConstrainedRationalFlatClassification,
) -> bool:
    """Verify a complete serialized rational-flat classification claim."""

    if claim.outcome.status != "COMPLETE_EXACT":
        return True
    try:
        return classify_clause_constrained_rational_flats(claim.problem) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def verify_rational_flat_representative(
    claim: ClauseConstrainedRationalFlatClassification,
    representative: RationalFlatOrbitRepresentative,
) -> bool:
    """Verify one representative against the retained bounded problem."""

    if claim.outcome.status != "COMPLETE_EXACT":
        return False
    try:
        expected = classify_clause_constrained_rational_flats(claim.problem)
        return (
            expected.outcome.status == "COMPLETE_EXACT"
            and representative in expected.outcome.representatives
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


__all__ = [
    "classify_clause_constrained_rational_flats",
    "verify_rational_flat_classification",
    "verify_rational_flat_representative",
]
