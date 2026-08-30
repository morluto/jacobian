"""Exact Gaussian elimination kernel for finite-field matrix rank.

Supports both prime and extension fields by working in the power-basis
coordinate representation used by FiniteFieldElement. Extension-field
elements are tuples of degree d; arithmetic is polynomial modular
arithmetic over the field presentation's irreducible modulus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sympy import Poly

from jacobian.math.finite_fields.values import (
    AxisBoundMatrix,
    FiniteFieldElement,
    FiniteFieldPresentation,
)


@dataclass(frozen=True, slots=True)
class MatrixRankData:
    """Private canonical data returned by the rank kernel."""

    rank: int
    pivot_rows: tuple[str, ...]
    pivot_columns: tuple[str, ...]


def _build_modulus(presentation: FiniteFieldPresentation) -> Poly:
    from sympy import Poly, symbols

    z = symbols("z")
    return Poly(
        sum(c * z**i for i, c in enumerate(presentation.modulus_coefficients)),
        z,
        modulus=presentation.characteristic,
    )


def _to_poly(element: FiniteFieldElement, z: Any, modulus: Poly) -> Poly:
    from sympy import Poly

    return Poly(
        sum(c * z**i for i, c in enumerate(element.coordinates)),
        z,
        modulus=element.presentation.characteristic,
    )


def _to_coordinates(poly: Poly, degree: int) -> tuple[int, ...]:
    return tuple(int(poly.nth(i)) % poly.domain.characteristic() for i in range(degree))


def _is_zero_poly(poly: Poly) -> bool:
    return bool(poly.is_zero)


def compute_matrix_rank(matrix: AxisBoundMatrix) -> MatrixRankData:
    """Compute exact rank via Gaussian elimination over the presented field.

    Works in the power-basis coordinate representation. For prime fields
    (degree 1), this reduces to ordinary modular Gaussian elimination.
    For extension fields, each entry is a polynomial modulo the irreducible
    modulus.
    """
    from sympy import invert, symbols

    presentation = matrix.presentation
    z = symbols("z")
    modulus = _build_modulus(presentation)
    rows = len(matrix.row_axis.labels)
    cols = len(matrix.column_axis.labels)

    # Convert entries to polynomial form.
    # poly_matrix[r][c] is a Poly in z mod modulus.
    poly_matrix: list[list[Poly]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            entry = matrix.entries[r][c]
            row.append(_to_poly(entry, z, modulus))
        poly_matrix.append(row)

    # Gaussian elimination with partial pivoting (first nonzero in column).
    rank = 0
    pivot_row_indices: list[int] = []
    pivot_col_indices: list[int] = []
    col = 0
    work_matrix = [list(row) for row in poly_matrix]  # mutable copy
    row_orig_indices = list(range(rows))

    while rank < rows and col < cols:
        # Find pivot in column col at or below row rank.
        pivot_row = None
        for r in range(rank, rows):
            if not _is_zero_poly(work_matrix[r][col]):
                pivot_row = r
                break

        if pivot_row is None:
            col += 1
            continue

        # Swap rows.
        if pivot_row != rank:
            work_matrix[rank], work_matrix[pivot_row] = (
                work_matrix[pivot_row],
                work_matrix[rank],
            )
            row_orig_indices[rank], row_orig_indices[pivot_row] = (
                row_orig_indices[pivot_row],
                row_orig_indices[rank],
            )

        # Eliminate below.
        pivot_val = work_matrix[rank][col]
        inv_pivot = invert(pivot_val, modulus)
        for r in range(rank + 1, rows):
            if not _is_zero_poly(work_matrix[r][col]):
                # Compute factor = work_matrix[r][col] / pivot_val
                # = work_matrix[r][col] * inverse(pivot_val) mod modulus
                factor = (work_matrix[r][col] * inv_pivot).rem(modulus)
                for c2 in range(col, cols):
                    work_matrix[r][c2] = (
                        work_matrix[r][c2] - factor * work_matrix[rank][c2]
                    ).rem(modulus)

        pivot_row_indices.append(row_orig_indices[rank])
        pivot_col_indices.append(col)
        rank += 1
        col += 1

    # Map internal indices back to axis labels.
    pivot_rows = tuple(matrix.row_axis.labels[i] for i in pivot_row_indices)
    pivot_columns = tuple(matrix.column_axis.labels[i] for i in pivot_col_indices)

    return MatrixRankData(
        rank=rank,
        pivot_rows=pivot_rows,
        pivot_columns=pivot_columns,
    )
