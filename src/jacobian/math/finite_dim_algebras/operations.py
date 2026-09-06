"""Exact finite-dimensional algebra operations."""

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_dim_algebras._models import CenterResult, StructureConstants


def _admit_center(algebra: StructureConstants) -> None:
    """Establish the prime-field and elimination contract for native callers."""

    from sympy import isprime

    if not isprime(algebra.field_order):
        raise OperationDomainValidationError(
            location=("algebra", "field_order"),
            code="finite_dim_algebra.field_order_not_prime",
            message="field_order must be prime",
        )


def _nullspace_mod_prime(
    entries: list[int], row_count: int, dimension: int, prime: int
) -> tuple[tuple[int, ...], ...]:
    from flint import nmod_mat

    matrix = nmod_mat(row_count, dimension, entries, prime)
    vectors, nullity = matrix.nullspace()
    return tuple(
        tuple(int(vectors[coordinate, basis]) for coordinate in range(dimension))
        for basis in range(nullity)
    )


def center_basis(algebra: StructureConstants) -> tuple[tuple[int, ...], ...]:
    """Return a canonical basis for the center over the declared prime field.

    Materialization and elimination-work bounds are request-scoped admission on
    ``StructureConstants``; this kernel executes the admitted algebra.
    """

    _admit_center(algebra)
    dimension = algebra.dimension
    prime = algebra.field_order
    multiplication = algebra.multiplication
    commutator_entries = [
        (
            multiplication[column][basis][coordinate]
            - multiplication[basis][column][coordinate]
        )
        % prime
        for basis in range(dimension)
        for coordinate in range(dimension)
        for column in range(dimension)
    ]
    return _nullspace_mod_prime(
        commutator_entries, dimension * dimension, dimension, prime
    )


def verify_center(claim: CenterResult) -> bool:
    """Check independence and equality with the full center, allowing any basis.

    Center admission bounds the commutator work by n^4. Two further matrices
    have at most 2n rows and n columns, within that same elimination envelope.
    """
    from flint import nmod_mat

    expected = center_basis(claim.algebra)
    if len(expected) != claim.center_dimension:
        return False
    dimension = claim.dimension
    prime = claim.algebra.field_order

    def rank(rows: tuple[tuple[int, ...], ...]) -> int:
        return int(
            nmod_mat(
                len(rows), dimension, [value for row in rows for value in row], prime
            ).rank()
        )

    return rank(claim.center_basis) == len(expected) and rank(
        (*expected, *claim.center_basis)
    ) == len(expected)


__all__ = ["center_basis", "verify_center"]
