"""Domain adapter for symbolic matrix operations."""

from __future__ import annotations

from jacobian.contracts.symbolic_matrix import (
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)
from jacobian.math.symbolic_matrix import (
    symbolic_characteristic_polynomial,
    symbolic_determinant,
    symbolic_eigenvalues,
    symbolic_rank,
)


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


def compute_symbolic_characteristic_polynomial(
    request: SymbolicMatrixRequest,
) -> SymbolicCharacteristicPolynomialResult:
    degree, coeffs = symbolic_characteristic_polynomial(
        [list(row) for row in request.matrix.entries],
        list(request.matrix.variables),
    )
    return SymbolicCharacteristicPolynomialResult(
        degree=degree,
        coefficients_descending=tuple(coeffs),
    )


def compute_symbolic_eigenvalues(
    request: SymbolicMatrixRequest,
) -> SymbolicEigenvaluesResult:
    eigenvalues = symbolic_eigenvalues(
        [list(row) for row in request.matrix.entries],
        list(request.matrix.variables),
    )
    return SymbolicEigenvaluesResult(
        eigenvalues=tuple(value for value, _ in eigenvalues),
        multiplicities=tuple(mult for _, mult in eigenvalues),
    )
