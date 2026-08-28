"""Private exact linear-algebra kernels for finite geometry."""

from __future__ import annotations

from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rref,
)


def rref_rank(matrix: list[list[int]], field_order: int) -> tuple[list[list[int]], int]:
    """Return reduced row echelon form and rank over the prime field."""

    shared_matrix = PrimeFieldMatrix(
        prime=field_order,
        entries=tuple(tuple(row) for row in matrix),
        columns=len(matrix[0]) if matrix else 0,
    )
    reduced, pivots = rref(shared_matrix)
    return [list(row) for row in reduced], len(pivots)


def canonical_basis(
    matrix: list[list[int]], field_order: int
) -> tuple[tuple[int, ...], ...]:
    """Return the unique RREF basis of the supplied row span."""

    reduced, rank = rref_rank(matrix, field_order)
    return tuple(tuple(row) for row in reduced[:rank])


def intersection_basis(
    basis_a: tuple[tuple[int, ...], ...],
    basis_b: tuple[tuple[int, ...], ...],
    field_order: int,
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the canonical basis of the intersection of two row spaces."""

    k, m = len(basis_a), len(basis_b)
    combined = [
        [
            *(basis_a[index][coordinate] % field_order for index in range(k)),
            *(-basis_b[index][coordinate] % field_order for index in range(m)),
        ]
        for coordinate in range(dimension)
    ]
    matrix = PrimeFieldMatrix(
        prime=field_order,
        entries=tuple(tuple(row) for row in combined),
        columns=k + m,
    )
    intersection_rows: list[list[int]] = []
    for coefficients in nullspace(matrix):
        intersection_rows.append(
            [
                sum(
                    coefficients[index] * basis_a[index][coordinate]
                    for index in range(k)
                )
                % field_order
                for coordinate in range(dimension)
            ]
        )
    return canonical_basis(intersection_rows, field_order) if intersection_rows else ()
