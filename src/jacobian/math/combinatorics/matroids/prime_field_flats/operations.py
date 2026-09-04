"""Native clause-constrained prime-field flat operations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.prime_field_flats._kernel import (
    classify_clause_constrained_prime_field_flats_kernel,
)
from jacobian.math.combinatorics.matroids.prime_field_flats._models import (
    ClauseConstrainedPrimeFieldFlatClassification,
    ClauseConstrainedPrimeFieldFlatProblem,
)


def classify_clause_constrained_prime_field_flats(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> ClauseConstrainedPrimeFieldFlatClassification:
    """Classify every admitted clause-constrained GF(p) flat modulo symmetry."""

    try:
        return classify_clause_constrained_prime_field_flats_kernel(problem)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("problem",),
            code=exc.type,
            message=exc.message(),
        ) from exc


__all__ = ["classify_clause_constrained_prime_field_flats"]
