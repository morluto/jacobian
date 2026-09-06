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
    PrimeFieldFlatOrbitRepresentative,
)


def classify_clause_constrained_prime_field_flats(
    problem: ClauseConstrainedPrimeFieldFlatProblem,
) -> ClauseConstrainedPrimeFieldFlatClassification:
    """Classify every admitted clause-constrained GF(p) flat modulo symmetry."""
    from sympy import isprime

    prime = problem.candidates.prime
    if not isprime(prime):
        raise OperationDomainValidationError(
            location=("problem", "candidates", "prime"),
            code="prime_field_flat.prime",
            message="prime must be a prime integer",
        )
    if problem.forbidden_vectors.prime != prime:
        raise OperationDomainValidationError(
            location=("problem", "forbidden_vectors", "prime"),
            code="prime_field_flat.matrix_prime",
            message="forbidden vectors must use the candidate configuration prime",
        )

    try:
        return classify_clause_constrained_prime_field_flats_kernel(problem)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("problem",),
            code=exc.type,
            message=exc.message(),
        ) from exc


def verify_prime_field_flat_classification(
    claim: ClauseConstrainedPrimeFieldFlatClassification,
) -> bool:
    """Verify a complete serialized prime-field flat classification claim."""

    if claim.outcome.status != "COMPLETE_EXACT":
        return True
    try:
        return classify_clause_constrained_prime_field_flats(claim.problem) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def verify_prime_field_flat_representative(
    claim: ClauseConstrainedPrimeFieldFlatClassification,
    representative: PrimeFieldFlatOrbitRepresentative,
) -> bool:
    """Verify one representative against the retained bounded problem."""

    if claim.outcome.status != "COMPLETE_EXACT":
        return False
    try:
        expected = classify_clause_constrained_prime_field_flats(claim.problem)
        return (
            expected.outcome.status == "COMPLETE_EXACT"
            and representative in expected.outcome.representatives
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


__all__ = [
    "classify_clause_constrained_prime_field_flats",
    "verify_prime_field_flat_classification",
    "verify_prime_field_flat_representative",
]
