"""Thin SymPy projections for exact matrix operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian.contracts.matrix_operations import (
    CharacteristicPolynomialResult,
    IntegerMatrixRequest,
    IntegerOutputMatrix,
    MatrixAdjugateResult,
    MatrixInverseResult,
    MatrixProductResult,
    MatrixTraceResult,
    NullspaceResult,
    OutputRational,
    RationalLinearSolveRequest,
    RationalLinearSolveResult,
    RationalMatrix,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RationalOutputMatrix,
    RrefResult,
    SmithNormalFormResult,
    SquareIntegerMatrixRequest,
    SquareRationalMatrixRequest,
)
from jacobian.math import matrices as native_matrices


def _rational(value: Any) -> OutputRational:
    fraction = Fraction(value)
    return OutputRational(
        num=str(fraction.numerator),
        den=str(fraction.denominator),
    )


def _qq_matrix(matrix: RationalMatrix) -> Any:
    import sympy

    return sympy.Matrix(
        [
            [sympy.Rational(int(value.num), int(value.den)) for value in row]
            for row in matrix.entries
        ]
    )


def compute_rref(request: RationalMatrixRequest) -> RrefResult:
    reduced, pivots = native_matrices.rref(_qq_matrix(request.matrix))
    columns = reduced.cols
    pivot_columns = tuple(int(column) for column in pivots)
    return RrefResult(
        reduced_matrix=RationalOutputMatrix(
            entries=tuple(
                tuple(_rational(reduced[row, column]) for column in range(columns))
                for row in range(reduced.rows)
            )
        ),
        rank=len(pivot_columns),
        pivot_columns=pivot_columns,
        free_columns=tuple(
            column for column in range(columns) if column not in pivot_columns
        ),
    )


def compute_nullspace(request: RationalMatrixRequest) -> NullspaceResult:
    import sympy

    matrix = _qq_matrix(request.matrix)
    reduced, pivots = matrix.rref()
    pivot_columns = tuple(int(column) for column in pivots)
    free_columns = tuple(
        column for column in range(matrix.cols) if column not in pivot_columns
    )
    pivot_row_by_column = {
        pivot_column: row for row, pivot_column in enumerate(pivot_columns)
    }
    basis: list[tuple[OutputRational, ...]] = []
    for free_column in free_columns:
        vector = [sympy.S.Zero] * matrix.cols
        vector[free_column] = sympy.S.One
        for pivot_column, row in pivot_row_by_column.items():
            vector[pivot_column] = -reduced[row, free_column]
        basis.append(tuple(_rational(value) for value in vector))
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
    polynomial = _qq_matrix(request.matrix).charpoly("lambda")
    return CharacteristicPolynomialResult(
        degree=polynomial.degree(),
        coefficients_descending=tuple(
            _rational(coefficient) for coefficient in polynomial.all_coeffs()
        ),
    )


def compute_smith_normal_form(
    request: IntegerMatrixRequest,
) -> SmithNormalFormResult:
    import sympy
    from sympy.matrices.normalforms import smith_normal_form

    source = sympy.Matrix(
        [[int(value) for value in row] for row in request.matrix.entries]
    )
    raw = smith_normal_form(source, domain=sympy.ZZ)
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
        normal_form=IntegerOutputMatrix(
            entries=tuple(
                tuple(str(int(canonical[row, column])) for column in range(raw.cols))
                for row in range(raw.rows)
            )
        ),
        rank=rank,
        invariant_factors=tuple(str(value) for value in invariant_factors),
    )


def compute_inverse(request: SquareIntegerMatrixRequest) -> MatrixInverseResult:
    import sympy

    source = sympy.Matrix(
        [[int(value) for value in row] for row in request.matrix.entries]
    )
    inverse = native_matrices.inverse(source)
    return MatrixInverseResult(
        inverse=RationalOutputMatrix(
            entries=tuple(
                tuple(_rational(inverse[row, column]) for column in range(inverse.cols))
                for row in range(inverse.rows)
            )
        )
    )


def compute_trace(request: SquareIntegerMatrixRequest) -> MatrixTraceResult:
    import sympy

    source = sympy.Matrix(
        [[int(value) for value in row] for row in request.matrix.entries]
    )
    return MatrixTraceResult(trace=str(int(native_matrices.trace(source))))


def compute_product(request: RationalMatrixProductRequest) -> MatrixProductResult:
    left = _qq_matrix(request.left)
    right = _qq_matrix(request.right)
    product = left * right
    return MatrixProductResult(
        product=RationalOutputMatrix(
            entries=tuple(
                tuple(_rational(product[row, column]) for column in range(product.cols))
                for row in range(product.rows)
            )
        ),
        left_rows=left.rows,
        inner_dimension=left.cols,
        right_columns=right.cols,
    )


def compute_rational_linear_solve(
    request: RationalLinearSolveRequest,
) -> RationalLinearSolveResult:
    import sympy

    source = _qq_matrix(request.matrix)
    rhs = sympy.Matrix(
        [sympy.Rational(int(value.num), int(value.den)) for value in request.rhs]
    )
    solution, parameters = source.gauss_jordan_solve(rhs)
    if parameters.rows:
        raise ValueError("linear system does not have a unique solution")
    return RationalLinearSolveResult(
        solution=tuple(_rational(value) for value in solution)
    )


def compute_adjugate(request: SquareIntegerMatrixRequest) -> MatrixAdjugateResult:
    import sympy

    source = sympy.Matrix(
        [[int(value) for value in row] for row in request.matrix.entries]
    )
    if source.rows != source.cols:
        raise ValueError("adjugate requires a square matrix")
    adjugate = source.adjugate()
    return MatrixAdjugateResult(
        adjugate=IntegerOutputMatrix(
            entries=tuple(
                tuple(
                    str(int(adjugate[row, column])) for column in range(adjugate.cols)
                )
                for row in range(adjugate.rows)
            )
        )
    )
