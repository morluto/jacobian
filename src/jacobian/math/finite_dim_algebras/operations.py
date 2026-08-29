"""Exact finite-dimensional algebra operations."""

from jacobian.math.finite_dim_algebras._models import (
    MAX_COMMUTATOR_ELIMINATION_WORK,
    MAX_COMMUTATOR_ENTRIES,
    StructureConstants,
    commutator_elimination_work,
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
    """Return a canonical basis for the center over the declared prime field."""

    dimension = algebra.dimension
    prime = algebra.field_order
    multiplication = algebra.multiplication
    if dimension**3 > MAX_COMMUTATOR_ENTRIES:
        raise ValueError("commutator matrix exceeds the materialization budget")
    if commutator_elimination_work(dimension) > MAX_COMMUTATOR_ELIMINATION_WORK:
        raise ValueError(
            "commutator nullspace exceeds the exact elimination-work budget"
        )
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
