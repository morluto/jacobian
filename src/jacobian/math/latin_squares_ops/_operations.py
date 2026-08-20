"""Domain functions for Latin square operations."""

from __future__ import annotations

from jacobian.math.latin_squares_ops._models import (
    LatinSquareCheckResult,
    LatinSquareRequest,
    LatinSquareTransposeResult,
    OrthogonalityRequest,
    OrthogonalityResult,
    TransposeRequest,
)


def compute_latin_square_check(request: LatinSquareRequest) -> LatinSquareCheckResult:
    """Check if a matrix is a Latin square."""
    n = request.square.order
    cells = request.square.cells
    for i in range(n):
        if len(set(cells[i])) != n:
            return LatinSquareCheckResult(is_latin=False)
        col = set()
        for j in range(n):
            col.add(cells[j][i])
        if len(col) != n:
            return LatinSquareCheckResult(is_latin=False)
    return LatinSquareCheckResult(is_latin=True)


def compute_orthogonality(request: OrthogonalityRequest) -> OrthogonalityResult:
    """Check if two Latin squares of the same order are orthogonal."""
    n = request.square_a.order
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            pair = (request.square_a.cells[i][j], request.square_b.cells[i][j])
            if pair in pairs:
                return OrthogonalityResult(is_orthogonal=False, pair_count=len(pairs))
            pairs.add(pair)
    return OrthogonalityResult(is_orthogonal=True, pair_count=len(pairs))


def compute_latin_square_transpose(
    request: TransposeRequest,
) -> LatinSquareTransposeResult:
    """Transpose a Latin square (swap rows and columns)."""
    n = request.square.order
    cells = request.square.cells
    transposed = tuple(tuple(cells[j][i] for j in range(n)) for i in range(n))
    return LatinSquareTransposeResult(transposed=transposed)
