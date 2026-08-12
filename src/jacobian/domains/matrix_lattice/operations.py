"""Thin SymPy projections for exact matrix operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.matrix_operations import (
    CharacteristicPolynomialResult,
    IntegerMatrixRequest,
    MatrixAdjugateResult,
    MatrixDeterminantRequest,
    MatrixDeterminantResult,
    MatrixInverseResult,
    MatrixProductResult,
    MatrixRankRequest,
    MatrixRankResult,
    MatrixTraceResult,
    NullspaceResult,
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RrefResult,
    SmithNormalFormResult,
    SquareIntegerMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.domains.matrix_lattice import conversions
from jacobian.math import matrices


def compute_determinant(
    request: MatrixDeterminantRequest,
) -> MatrixDeterminantResult:
    determinant = matrices.determinant(
        conversions.rational_matrix_to_sympy(request.matrix)
    )
    return MatrixDeterminantResult(
        determinant=conversions.rational_from_sympy(determinant)
    )


def compute_rank(request: MatrixRankRequest) -> MatrixRankResult:
    rank, pivot_columns = matrices.rank(
        conversions.rational_matrix_to_sympy(request.matrix)
    )
    return MatrixRankResult(rank=rank, pivot_columns=pivot_columns)


def compute_rref(request: RationalMatrixRequest) -> RrefResult:
    reduced, pivots = matrices.rref(
        conversions.rational_matrix_to_sympy(request.matrix)
    )
    columns = reduced.cols
    pivot_columns = tuple(int(column) for column in pivots)
    return RrefResult(
        reduced_matrix=conversions.rational_matrix_from_sympy(reduced),
        rank=len(pivot_columns),
        pivot_columns=pivot_columns,
        free_columns=tuple(
            column for column in range(columns) if column not in pivot_columns
        ),
    )


def compute_nullspace(request: RationalMatrixRequest) -> NullspaceResult:
    import sympy

    matrix = conversions.rational_matrix_to_sympy(request.matrix)
    reduced, pivots = matrices.rref(matrix)
    pivot_columns = tuple(int(column) for column in pivots)
    free_columns = tuple(
        column for column in range(matrix.cols) if column not in pivot_columns
    )
    pivot_row_by_column = {
        pivot_column: row for row, pivot_column in enumerate(pivot_columns)
    }
    basis: list[tuple[CanonicalRational, ...]] = []
    for free_column in free_columns:
        vector = [sympy.S.Zero] * matrix.cols
        vector[free_column] = sympy.S.One
        for pivot_column, row in pivot_row_by_column.items():
            vector[pivot_column] = -reduced[row, free_column]
        basis.append(tuple(conversions.rational_from_sympy(value) for value in vector))
    return NullspaceResult(
        ambient_dimension=matrix.cols,
        rank=len(pivot_columns),
        nullity=len(basis),
        basis_vectors=tuple(basis),
        free_columns=free_columns,
    )


def compute_characteristic_polynomial(
    request: SquareRationalMatrixRequest,
) -> CharacteristicPolynomialResult:
    polynomial = matrices.characteristic_polynomial(
        conversions.rational_matrix_to_sympy(request.matrix), "lambda"
    )
    return CharacteristicPolynomialResult(
        degree=polynomial.degree(),
        coefficients_descending=tuple(
            conversions.rational_from_sympy(coefficient)
            for coefficient in polynomial.all_coeffs()
        ),
    )


def compute_smith_normal_form(
    request: IntegerMatrixRequest,
) -> SmithNormalFormResult:
    import sympy

    raw = matrices.smith_normal_form(
        conversions.integer_matrix_to_sympy(request.matrix)
    )
    diagonal_count = min(raw.rows, raw.cols)
    diagonal = tuple(int(raw[index, index]) for index in range(diagonal_count))
    rank = next(
        (index for index, value in enumerate(diagonal) if value == 0),
        diagonal_count,
    )
    if any(diagonal[index] != 0 for index in range(rank, diagonal_count)):
        raise RuntimeError("Smith backend returned a nonzero factor after a zero")
    invariant_factors = tuple(abs(value) for value in diagonal[:rank])
    canonical = sympy.zeros(raw.rows, raw.cols)
    for index, value in enumerate(invariant_factors):
        canonical[index, index] = value
    return SmithNormalFormResult(
        normal_form=conversions.integer_matrix_from_sympy(canonical),
        rank=rank,
        invariant_factors=tuple(
            format_canonical_integer(value) for value in invariant_factors
        ),
    )


def compute_inverse(request: SquareIntegerMatrixRequest) -> MatrixInverseResult:
    inverse = matrices.inverse(conversions.integer_matrix_to_sympy(request.matrix))
    return MatrixInverseResult(inverse=conversions.rational_matrix_from_sympy(inverse))


def compute_trace(request: SquareIntegerMatrixRequest) -> MatrixTraceResult:
    return MatrixTraceResult(
        trace=format_canonical_integer(
            matrices.trace(conversions.integer_matrix_to_sympy(request.matrix))
        )
    )


def compute_product(request: RationalMatrixProductRequest) -> MatrixProductResult:
    left = conversions.rational_matrix_to_sympy(request.left)
    right = conversions.rational_matrix_to_sympy(request.right)
    product = matrices.multiply(left, right)
    return MatrixProductResult(
        product=conversions.rational_matrix_from_sympy(product),
        left_rows=left.rows,
        inner_dimension=left.cols,
        right_columns=right.cols,
    )


def compute_rational_linear_solve(
    request: RationalLinearSolveRequest,
) -> RationalLinearSolveResult:
    import sympy

    source = conversions.rational_matrix_to_sympy(request.matrix)
    rhs = sympy.Matrix([sympy.Rational(value.as_fraction()) for value in request.rhs])
    solution, parameters = matrices.solve_linear_system(source, rhs)
    if parameters.rows:
        raise ValueError("linear system does not have a unique solution")
    return RationalLinearSolveResult(
        solution=tuple(conversions.rational_from_sympy(value) for value in solution)
    )


def compute_adjugate(request: SquareIntegerMatrixRequest) -> MatrixAdjugateResult:
    adjugate = matrices.adjugate(conversions.integer_matrix_to_sympy(request.matrix))
    return MatrixAdjugateResult(
        adjugate=conversions.integer_matrix_from_sympy(adjugate)
    )
