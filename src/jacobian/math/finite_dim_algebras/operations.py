"""Exact finite-dimensional algebra operations."""

from jacobian.math.finite_dim_algebras._models import StructureConstants


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


__all__ = ["center_basis"]
