"""Exact finite-field matrix rank kernel backed by a maintained library.

Rank and pivot tracking use python-flint's maintained ``fq_default`` element
arithmetic instead of constructing SymPy polynomials.  For prime fields the
backend degenerates to modular integer arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jacobian.math.finite_fields._flint import context as _backend_context
from jacobian.math.finite_fields._flint import to_backend as _to_backend
from jacobian.math.finite_fields.values import AxisBoundMatrix


@dataclass(frozen=True, slots=True)
class MatrixRankData:
    """Private canonical data returned by the rank kernel."""

    rank: int
    pivot_rows: tuple[str, ...]
    pivot_columns: tuple[str, ...]


def compute_matrix_rank(
    matrix: AxisBoundMatrix,
    *,
    execution_checkpoint: Callable[[str], None] | None = None,
) -> MatrixRankData:
    """Compute exact rank and pivot labels over the presented finite field.

    Delegates all finite-field arithmetic to python-flint's maintained
    ``fq_default`` backend rather than constructing SymPy polynomials.
    """

    checkpoint = execution_checkpoint or (lambda _stage: None)
    checkpoint("before finite-field context construction")
    presentation = matrix.presentation
    rows = len(matrix.row_axis.labels)
    cols = len(matrix.column_axis.labels)

    if rows == 0 or cols == 0:
        return MatrixRankData(rank=0, pivot_rows=(), pivot_columns=())

    active_context = _backend_context(presentation)
    zero = active_context.zero()

    # Convert entries to maintained backend elements once.
    entries: list[list[Any]] = []
    for r in range(rows):
        checkpoint(f"before finite-field row {r} conversion")
        row = []
        for c in range(cols):
            row.append(_to_backend(matrix.entries[r][c], active_context=active_context))
        entries.append(row)
        checkpoint(f"after finite-field row {r} conversion")

    # Gaussian elimination using the maintained backend's arithmetic.
    pivot_row_indices: list[int] = []
    pivot_col_indices: list[int] = []
    rank = 0
    col = 0
    row_perm = list(range(rows))

    while rank < rows and col < cols:
        checkpoint(f"before finite-field elimination column {col}")
        pivot_row = None
        for r in range(rank, rows):
            if entries[r][col] != zero:
                pivot_row = r
                break

        if pivot_row is None:
            col += 1
            continue

        if pivot_row != rank:
            entries[rank], entries[pivot_row] = (
                entries[pivot_row],
                entries[rank],
            )
            row_perm[rank], row_perm[pivot_row] = (
                row_perm[pivot_row],
                row_perm[rank],
            )

        pivot_val = entries[rank][col]
        for r in range(rank + 1, rows):
            if entries[r][col] != zero:
                factor = entries[r][col] / pivot_val
                for c2 in range(col, cols):
                    entries[r][c2] = entries[r][c2] - factor * entries[rank][c2]

        pivot_row_indices.append(row_perm[rank])
        pivot_col_indices.append(col)
        rank += 1
        col += 1
        checkpoint(f"after finite-field elimination column {col - 1}")

    # The public result exposes pivot row and column sets, not their elimination
    # pairing. Canonicalize each set independently to satisfy both axis-order
    # contracts after row swaps.
    pivot_rows = tuple(matrix.row_axis.labels[i] for i in sorted(pivot_row_indices))
    pivot_columns = tuple(
        matrix.column_axis.labels[i] for i in sorted(pivot_col_indices)
    )

    return MatrixRankData(
        rank=rank,
        pivot_rows=pivot_rows,
        pivot_columns=pivot_columns,
    )
