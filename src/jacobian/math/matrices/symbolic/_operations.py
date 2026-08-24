"""Domain-owned symbolic matrix operations."""

from __future__ import annotations

from sympy.matrices.exceptions import MatrixError

from jacobian.math.matrices.symbolic import (
    symbolic_characteristic_polynomial,
    symbolic_determinant,
    symbolic_eigenvalues,
    symbolic_matrix_multiply,
    symbolic_rank,
)
from jacobian.math.matrices.symbolic._models import (
    SymbolicCharacteristicPolynomialRequest,
    SymbolicCharacteristicPolynomialResult,
    SymbolicDeterminantRequest,
    SymbolicDeterminantResult,
    SymbolicEigenvaluesResult,
    SymbolicLinearSystemRequest,
    SymbolicLinearSystemResult,
    SymbolicMatrix,
    SymbolicMatrixProductRequest,
    SymbolicMatrixRequest,
    SymbolicRankResult,
)


def compute_symbolic_determinant(
    request: SymbolicDeterminantRequest,
) -> SymbolicDeterminantResult:
    determinant = symbolic_determinant(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicDeterminantResult(determinant=determinant)


def compute_symbolic_rank(
    request: SymbolicMatrixRequest,
) -> SymbolicRankResult:
    rank, pivot_columns = symbolic_rank(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicRankResult(rank=rank, pivot_columns=pivot_columns)


def compute_symbolic_matrix_product(
    request: SymbolicMatrixProductRequest,
) -> SymbolicMatrix:
    """Compute one exact symbolic matrix product."""

    return symbolic_matrix_multiply(request.left, request.right)


def compute_symbolic_characteristic_polynomial(
    request: SymbolicCharacteristicPolynomialRequest,
) -> SymbolicCharacteristicPolynomialResult:
    degree, coeffs = symbolic_characteristic_polynomial(
        request.matrix.entries,
        request.matrix.variables,
    )
    return SymbolicCharacteristicPolynomialResult(
        degree=degree,
        coefficients_descending=tuple(coeffs),
    )


def compute_symbolic_eigenvalues(
    request: SymbolicCharacteristicPolynomialRequest,
) -> SymbolicEigenvaluesResult:
    entries = request.matrix.entries
    variables = request.matrix.variables
    try:
        eigenvalues = symbolic_eigenvalues(entries, variables)
    except MatrixError:
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


def compute_symbolic_linear_system(
    request: SymbolicLinearSystemRequest,
) -> SymbolicLinearSystemResult:
    """Classify and solve ``A x = b`` over ``QQ(t_1, ..., t_n)``."""

    from jacobian.math.matrices.symbolic import symbolic_linear_system_solve
    from jacobian.math.matrices.symbolic._models import (
        SymbolicLinearSystemResult,
    )

    classification, solution, particular, nullspace = symbolic_linear_system_solve(
        request.matrix.entries,
        request.rhs,
        request.matrix.variables,
    )

    return SymbolicLinearSystemResult(
        system=request,
        classification=classification,
        solution=solution,
        particular_solution=particular,
        nullspace_basis=nullspace,
    )
