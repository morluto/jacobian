"""Domain-owned symbolic matrix operations."""

from __future__ import annotations

from sympy.matrices.exceptions import MatrixError

from jacobian.math.matrices.symbolic import (
    symbolic_characteristic_polynomial,
    symbolic_determinant,
    symbolic_eigenvalues,
    symbolic_linear_system_solve,
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
from jacobian.math.polynomials._conversions import rational_function_to_sympy


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

    return SymbolicLinearSystemResult._from_kernel(
        system=request,
        classification=classification,
        solution=solution,
        particular_solution=particular,
        nullspace_basis=nullspace,
    )


def verify_symbolic_linear_system_result(result: SymbolicLinearSystemResult) -> bool:
    """Verify one independently supplied symbolic-system result.

    The retained request was admitted before result parsing.  Replaying it is
    therefore bounded; this explicit path keeps SymPy execution out of the
    transport model while preserving source-bound claim verification.
    """

    expected_classification, expected_solution, _particular, _nullspace = (
        symbolic_linear_system_solve(
            result.system.matrix.entries,
            result.system.rhs,
            result.system.matrix.variables,
        )
    )
    if result.classification != expected_classification:
        return False
    if result.classification == "INCONSISTENT":
        return True
    if result.classification == "UNIQUE":
        return result.solution == expected_solution

    particular = result.particular_solution
    if particular is None:
        return False
    import sympy

    coefficient = sympy.Matrix(
        [
            [rational_function_to_sympy(entry) for entry in row]
            for row in result.system.matrix.entries
        ]
    )
    rhs = sympy.Matrix(
        [[rational_function_to_sympy(value)] for value in result.system.rhs]
    )
    particular_vector = sympy.Matrix(
        [[rational_function_to_sympy(value)] for value in particular]
    )
    if any(sympy.cancel(entry) != 0 for entry in coefficient * particular_vector - rhs):
        return False

    basis = result.nullspace_basis or ()
    kernel_columns = []
    for vector in basis:
        column = sympy.Matrix([[rational_function_to_sympy(value)] for value in vector])
        if any(sympy.cancel(entry) != 0 for entry in coefficient * column):
            return False
        kernel_columns.append(column)
    nullity = coefficient.cols - coefficient.rank()
    return len(kernel_columns) == nullity and (
        not nullity or sympy.Matrix.hstack(*kernel_columns).rank() == nullity
    )
