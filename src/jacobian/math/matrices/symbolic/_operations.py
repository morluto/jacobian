"""Domain-owned symbolic matrix operations."""

from __future__ import annotations

from jacobian.math.matrices.symbolic import (
    symbolic_characteristic_polynomial,
    symbolic_determinant,
    symbolic_eigenvalues,
    symbolic_rank,
)
from jacobian.math.matrices.symbolic._models import (
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicMatrixRequest,
    SymbolicRankResult,
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
    entries = [list(row) for row in request.matrix.entries]
    variables = list(request.matrix.variables)
    try:
        eigenvalues = symbolic_eigenvalues(entries, variables)
    except Exception:
        # SymPy raises MatrixError when eigenvalues cannot be represented
        # in radicals.  Return the exact characteristic polynomial instead.
        degree, coeffs = symbolic_characteristic_polynomial(entries, variables)
        return SymbolicEigenvaluesResult(
            representation="ROOTS_BY_POLYNOMIAL",
            characteristic_polynomial=tuple(coeffs),
            degree=degree,
        )
    return SymbolicEigenvaluesResult(
        representation="EXPLICIT_ROOTS",
        eigenvalues=tuple(value for value, _ in eigenvalues),
        multiplicities=tuple(mult for _, mult in eigenvalues),
    )
