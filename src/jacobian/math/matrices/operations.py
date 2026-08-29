"""Exact matrix operations on canonical SymPy matrix inputs.

This is the supported public API for ``jacobian.math.matrices``. Private tool
declarations convert wire models to SymPy matrices, call these functions, and
convert results back. The SymPy backend is private to this module and loaded
lazily so importing ``jacobian.math`` does not eagerly load packaged backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices import _conversions as conversions
from jacobian.math.matrices._operation_models import (
    MAX_DETERMINANT_MATRIX_DIMENSION,
    MAX_INPUT_SCALAR_DIGITS,
    MAX_KRONECKER_PRODUCT_AXIS,
    MAX_PERMANENT_RYSER_SUBSETS,
    CharacteristicPolynomialResult,
    MatrixAdjugateResult,
    MatrixDeterminantResult,
    MatrixInverseResult,
    MatrixKroneckerProductResult,
    MatrixPartialTraceResult,
    MatrixPermanentResult,
    MatrixProductResult,
    MatrixRankResult,
    MatrixTraceResult,
    NullspaceResult,
    RationalLinearSolveResult,
    RrefResult,
    _require_computation_dimensions,
    _require_square_system_admission,
    _validation_error,
)
from jacobian.math.matrices.values import IntegerMatrix, RationalMatrix, SmithNormalForm

if TYPE_CHECKING:
    from sympy.matrices.matrixbase import MatrixBase

__all__ = [
    "adjugate",
    "adjugate_result",
    "characteristic_polynomial",
    "characteristic_polynomial_result",
    "determinant",
    "determinant_result",
    "inverse",
    "inverse_result",
    "kronecker_product",
    "kronecker_product_result",
    "multiply",
    "nullspace_result",
    "partial_trace",
    "partial_trace_result",
    "permanent",
    "permanent_result",
    "product_result",
    "rank",
    "rank_result",
    "rational_linear_solve_result",
    "rref",
    "rref_result",
    "smith_normal_form",
    "smith_normal_form_result",
    "solve_linear_system",
    "trace",
    "trace_result",
]


class MatrixSingularError(ValueError):
    """The exact inverse kernel proved that a matrix has no inverse."""


def _exact_matrix(value: MatrixBase, *, maximum_dimension: int = 32) -> MatrixBase:
    import sympy
    from sympy.matrices.matrixbase import MatrixBase

    if not isinstance(value, MatrixBase):
        raise TypeError("matrix must be a SymPy MatrixBase")
    if (
        not 1 <= value.rows <= maximum_dimension
        or not 1 <= value.cols <= maximum_dimension
    ):
        raise ValueError(f"matrix dimensions must be between 1 and {maximum_dimension}")
    if any(not entry.is_number or entry.is_finite is not True for entry in value):
        raise ValueError("matrix entries must be finite exact numbers")
    if any(entry.has(sympy.Float) for entry in value):
        raise ValueError("matrix entries must be exact; SymPy Float is not supported")
    return value


def rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    reduced, pivots = _exact_matrix(matrix).rref()
    return reduced, tuple(int(pivot) for pivot in pivots)


def inverse(matrix: MatrixBase) -> MatrixBase:
    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("inverse requires a square matrix")
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    try:
        numerator, denominator = DomainMatrix.from_Matrix(source).inv_den()
    except DMNonInvertibleMatrixError as exc:
        raise MatrixSingularError("matrix is singular; inverse does not exist") from exc
    return numerator.to_Matrix() / int(denominator)


def trace(matrix: MatrixBase) -> Any:
    import sympy

    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("trace requires a square matrix")
    return sympy.simplify(source.trace())


def characteristic_polynomial(matrix: MatrixBase, variable: str) -> Any:
    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    return source.charpoly(variable)


def determinant(matrix: MatrixBase) -> Any:
    source = _exact_matrix(matrix, maximum_dimension=64)
    if source.rows != source.cols:
        raise ValueError("determinant requires a square matrix")
    return source.det(method="bareiss")


def rank(matrix: MatrixBase) -> tuple[int, tuple[int, ...]]:
    _, pivots = rref(matrix)
    return len(pivots), pivots


def smith_normal_form(matrix: MatrixBase) -> MatrixBase:
    import sympy
    from sympy.matrices.normalforms import smith_normal_form as sympy_smith_normal_form

    return sympy_smith_normal_form(_exact_matrix(matrix), domain=sympy.ZZ)


def multiply(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    return _exact_matrix(left) * _exact_matrix(right)


def solve_linear_system(
    matrix: MatrixBase,
    right_hand_side: MatrixBase,
) -> tuple[MatrixBase, MatrixBase]:
    return cast(
        tuple[Any, Any],
        _exact_matrix(matrix).gauss_jordan_solve(_exact_matrix(right_hand_side)),
    )


def adjugate(matrix: MatrixBase) -> MatrixBase:
    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("adjugate requires a square matrix")
    return source.adjugate()


def kronecker_product(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    import sympy

    return sympy.kronecker_product(_exact_matrix(left), _exact_matrix(right))


def partial_trace(
    matrix: MatrixBase,
    traced_dimension: int,
    kept_dimension: int,
) -> MatrixBase:
    """Trace out the first traced_dimension factor of a Kronecker product.

    The composite matrix is the Kronecker product A (x) B of a
    traced_dimension x traced_dimension matrix A (the traced subsystem)
    by a kept_dimension x kept_dimension matrix B (the kept subsystem),
    stored as a block matrix in row-major block order.  The returned matrix is
    the trace over the traced factor, i.e. trace(A) * B.
    """

    import sympy

    source = _exact_matrix(matrix)
    total = traced_dimension * kept_dimension
    if source.rows != source.cols:
        raise ValueError("partial trace requires a square composite matrix")
    if source.rows != total:
        raise ValueError(
            "partial trace dimensions are inconsistent with the composite matrix"
        )
    if traced_dimension <= 0 or kept_dimension <= 0:
        raise ValueError("partial trace subsystem dimensions must be positive")
    accumulator = sympy.zeros(kept_dimension)
    for block in range(traced_dimension):
        block_row = block * kept_dimension
        block_col = block * kept_dimension
        accumulator = sympy.Matrix(
            [
                [
                    accumulator[i, j] + source[block_row + i, block_col + j]
                    for j in range(kept_dimension)
                ]
                for i in range(kept_dimension)
            ]
        )
    return accumulator


def permanent(matrix: MatrixBase) -> Any:
    from sympy import Permanent

    source = _exact_matrix(matrix, maximum_dimension=64)
    if source.rows != source.cols:
        raise ValueError("permanent requires a square matrix")
    return Permanent(source).doit()


def _admit(
    check: Callable[..., None], *values: Any, location: tuple[str, ...] = ("matrix",)
) -> None:
    try:
        check(*values)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


def _admit_rational_matrix(matrix: RationalMatrix) -> None:
    _require_computation_dimensions(matrix.entries)
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    require_matrix_scalar_digits(
        matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
    )


def _admit_rational(matrix: RationalMatrix) -> None:
    _admit_rational_matrix(matrix)


def _admit_square_rational(matrix: RationalMatrix) -> None:
    _admit_rational_matrix(matrix)
    if len(matrix.entries) != len(matrix.entries[0]):
        raise _validation_error("budget_exceeded", "operation requires a square matrix")


def _admit_integer(matrix: IntegerMatrix) -> None:
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    require_matrix_scalar_digits(
        matrix.entries,
        maximum=MAX_INPUT_SCALAR_DIGITS,
        label="matrix input",
    )


def _admit_square_integer(matrix: IntegerMatrix) -> None:
    _admit_integer(matrix)
    rows = len(matrix.entries)
    if rows == 0 or rows != len(matrix.entries[0]):
        raise _validation_error(
            "budget_exceeded", "operation requires a square integer matrix"
        )


def _admit_permanent(matrix: RationalMatrix) -> None:
    _admit_rational_matrix(matrix)
    order = len(matrix.entries)
    if order != len(matrix.entries[0]):
        raise _validation_error(
            "budget_exceeded", "permanent computation requires a square matrix"
        )
    if (1 << order) > MAX_PERMANENT_RYSER_SUBSETS:
        raise _validation_error(
            "budget_exceeded",
            "permanent computation exceeds the "
            f"{MAX_PERMANENT_RYSER_SUBSETS}-subset Ryser work budget",
        )


def _admit_product(left: RationalMatrix, right: RationalMatrix) -> None:
    if len(left.entries[0]) != len(right.entries):
        raise _validation_error(
            "budget_exceeded",
            "matrix multiplication requires the left column count to equal the right row count",
        )
    _admit_rational_matrix(left)
    _admit_rational_matrix(right)


def _admit_kronecker(left: RationalMatrix, right: RationalMatrix) -> None:
    _admit_rational_matrix(left)
    _admit_rational_matrix(right)
    if (
        len(left.entries) * len(right.entries) > MAX_KRONECKER_PRODUCT_AXIS
        or len(left.entries[0]) * len(right.entries[0]) > MAX_KRONECKER_PRODUCT_AXIS
    ):
        raise _validation_error(
            "budget_exceeded",
            "kronecker products must fit within "
            f"{MAX_KRONECKER_PRODUCT_AXIS} rows and columns",
        )


def _admit_partial_trace(
    matrix: RationalMatrix, traced_dimension: int, kept_dimension: int
) -> None:
    _admit_rational_matrix(matrix)
    total = traced_dimension * kept_dimension
    if len(matrix.entries) != total or len(matrix.entries[0]) != total:
        raise _validation_error(
            "budget_exceeded",
            "composite matrix must be square of order traced_dimension * kept_dimension",
        )


def _admit_determinant(matrix: RationalMatrix) -> None:
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    require_matrix_scalar_digits(
        matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
    )
    if len(matrix.entries) != len(matrix.entries[0]):
        raise _validation_error("budget_exceeded", "operation requires a square matrix")
    if len(matrix.entries) > MAX_DETERMINANT_MATRIX_DIMENSION:
        raise _validation_error(
            "budget_exceeded",
            "determinant matrices are limited to order "
            f"{MAX_DETERMINANT_MATRIX_DIMENSION}",
        )


def _admit_linear_solve(
    matrix: RationalMatrix, rhs: tuple[CanonicalRational, ...]
) -> None:
    _require_square_system_admission(matrix, rhs)


def determinant_result(matrix: RationalMatrix) -> MatrixDeterminantResult:
    _admit(_admit_determinant, matrix)
    value = determinant(conversions.rational_matrix_to_sympy(matrix))
    return MatrixDeterminantResult(determinant=conversions.rational_from_sympy(value))


def rank_result(matrix: RationalMatrix) -> MatrixRankResult:
    _admit(_admit_rational, matrix)
    value, pivot_columns = rank(conversions.rational_matrix_to_sympy(matrix))
    return MatrixRankResult._from_kernel(
        matrix=matrix,
        rank=value,
        pivot_columns=tuple(int(column) for column in pivot_columns),
    )


def rref_result(matrix: RationalMatrix) -> RrefResult:
    _admit(_admit_rational, matrix)
    reduced, pivots = rref(conversions.rational_matrix_to_sympy(matrix))
    columns = reduced.cols
    pivot_columns = tuple(int(column) for column in pivots)
    return RrefResult._from_kernel(
        matrix=matrix,
        reduced_matrix=conversions.rational_matrix_from_sympy(reduced),
        rank=len(pivot_columns),
        pivot_columns=pivot_columns,
        free_columns=tuple(
            column for column in range(columns) if column not in pivot_columns
        ),
    )


def nullspace_result(matrix: RationalMatrix) -> NullspaceResult:
    _admit(_admit_rational, matrix)
    import sympy

    source = conversions.rational_matrix_to_sympy(matrix)
    reduced, pivots = rref(source)
    pivot_columns = tuple(int(column) for column in pivots)
    free_columns = tuple(
        column for column in range(source.cols) if column not in pivot_columns
    )
    pivot_row_by_column = {
        pivot_column: row for row, pivot_column in enumerate(pivot_columns)
    }
    basis: list[tuple[CanonicalRational, ...]] = []
    for free_column in free_columns:
        vector = [sympy.S.Zero] * source.cols
        vector[free_column] = sympy.S.One
        for pivot_column, row in pivot_row_by_column.items():
            vector[pivot_column] = -reduced[row, free_column]
        basis.append(tuple(conversions.rational_from_sympy(value) for value in vector))
    return NullspaceResult._from_kernel(
        matrix=matrix,
        ambient_dimension=source.cols,
        rank=len(pivot_columns),
        nullity=len(basis),
        basis_vectors=tuple(basis),
        free_columns=free_columns,
    )


def characteristic_polynomial_result(
    matrix: RationalMatrix,
) -> CharacteristicPolynomialResult:
    _admit(_admit_square_rational, matrix)
    polynomial = characteristic_polynomial(
        conversions.rational_matrix_to_sympy(matrix), "lambda"
    )
    return CharacteristicPolynomialResult(
        degree=polynomial.degree(),
        coefficients_descending=tuple(
            conversions.rational_from_sympy(coefficient)
            for coefficient in polynomial.all_coeffs()
        ),
    )


def smith_normal_form_result(matrix: IntegerMatrix) -> SmithNormalForm:
    _admit(_admit_integer, matrix)
    raw = smith_normal_form(conversions.integer_matrix_to_sympy(matrix))
    return conversions.smith_normal_form_from_sympy(raw)


def inverse_result(matrix: IntegerMatrix) -> MatrixInverseResult:
    _admit(_admit_square_integer, matrix)
    try:
        value = inverse(conversions.integer_matrix_to_sympy(matrix))
    except MatrixSingularError as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="matrix.singular_matrix",
            message="matrix is singular; inverse does not exist",
        ) from exc
    return MatrixInverseResult(inverse=conversions.rational_matrix_from_sympy(value))


def trace_result(matrix: IntegerMatrix) -> MatrixTraceResult:
    _admit(_admit_square_integer, matrix)
    return MatrixTraceResult(
        trace=format_canonical_integer(
            trace(conversions.integer_matrix_to_sympy(matrix))
        )
    )


def product_result(left: RationalMatrix, right: RationalMatrix) -> MatrixProductResult:
    _admit(_admit_product, left, right, location=("left", "right"))
    left_source = conversions.rational_matrix_to_sympy(left)
    right_source = conversions.rational_matrix_to_sympy(right)
    product = multiply(left_source, right_source)
    return MatrixProductResult(
        product=conversions.rational_matrix_from_sympy(product),
        left_rows=left_source.rows,
        inner_dimension=left_source.cols,
        right_columns=right_source.cols,
    )


def rational_linear_solve_result(
    matrix: RationalMatrix, rhs: tuple[CanonicalRational, ...]
) -> RationalLinearSolveResult:
    _admit(_admit_linear_solve, matrix, rhs, location=("matrix", "rhs"))
    import sympy

    source = conversions.rational_matrix_to_sympy(matrix)
    rhs_source = sympy.Matrix([sympy.Rational(value.as_fraction()) for value in rhs])
    try:
        solution, parameters = solve_linear_system(source, rhs_source)
    except ValueError:
        return RationalLinearSolveResult._from_kernel(
            matrix=matrix,
            rhs=rhs,
            outcome="INCONSISTENT",
        )
    if parameters.rows:
        return RationalLinearSolveResult._from_kernel(
            matrix=matrix,
            rhs=rhs,
            outcome="NON_UNIQUE",
        )
    return RationalLinearSolveResult._from_kernel(
        matrix=matrix,
        rhs=rhs,
        outcome="UNIQUE",
        solution=tuple(conversions.rational_from_sympy(value) for value in solution),
    )


def adjugate_result(matrix: IntegerMatrix) -> MatrixAdjugateResult:
    _admit(_admit_square_integer, matrix)
    value = adjugate(conversions.integer_matrix_to_sympy(matrix))
    return MatrixAdjugateResult(adjugate=conversions.integer_matrix_from_sympy(value))


def permanent_result(matrix: RationalMatrix) -> MatrixPermanentResult:
    _admit(_admit_permanent, matrix)
    value = permanent(conversions.rational_matrix_to_sympy(matrix))
    return MatrixPermanentResult(
        permanent=conversions.rational_from_sympy(value),
    )


def kronecker_product_result(
    left: RationalMatrix, right: RationalMatrix
) -> MatrixKroneckerProductResult:
    _admit(_admit_kronecker, left, right, location=("left", "right"))
    left_source = conversions.rational_matrix_to_sympy(left)
    right_source = conversions.rational_matrix_to_sympy(right)
    product = kronecker_product(left_source, right_source)
    return MatrixKroneckerProductResult(
        product=conversions.rational_matrix_from_sympy(product),
        left_rows=left_source.rows,
        left_columns=left_source.cols,
        right_rows=right_source.rows,
        right_columns=right_source.cols,
    )


def partial_trace_result(
    matrix: RationalMatrix, traced_dimension: int, kept_dimension: int
) -> MatrixPartialTraceResult:
    _admit(
        _admit_partial_trace,
        matrix,
        traced_dimension,
        kept_dimension,
        location=("matrix", "traced_dimension", "kept_dimension"),
    )
    source = conversions.rational_matrix_to_sympy(matrix)
    reduced = partial_trace(
        source,
        traced_dimension,
        kept_dimension,
    )
    return MatrixPartialTraceResult(
        reduced_matrix=conversions.rational_matrix_from_sympy(reduced),
        traced_dimension=traced_dimension,
        kept_dimension=kept_dimension,
    )
