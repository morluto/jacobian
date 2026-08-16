"""Domain adapter for symbolic matrix operations."""

from __future__ import annotations

from jacobian.contracts.symbolic_matrix import (
    SymbolicDeterminantResult,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.symbolic_matrix import symbolic_determinant, symbolic_rank


def compute_symbolic_determinant(
    request: SymbolicMatrixRequest,
) -> SymbolicDeterminantResult:
    determinant = symbolic_determinant(
        [list(row) for row in request.matrix.entries],
        list(request.matrix.variables),
    )
    return SymbolicDeterminantResult(determinant=determinant)


def compute_symbolic_rank(
    request: SymbolicMatrixRequest,
) -> SymbolicRankResult:
    rank, pivot_columns = symbolic_rank(
        [list(row) for row in request.matrix.entries],
        list(request.matrix.variables),
    )
    return SymbolicRankResult(rank=rank, pivot_columns=pivot_columns)
