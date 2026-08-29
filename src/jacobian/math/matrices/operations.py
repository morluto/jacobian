"""Exact matrix operations on canonical SymPy matrix inputs.

This is the supported public API for ``jacobian.math.matrices``. Private tool
declarations convert wire models to SymPy matrices, call these functions, and
convert results back. The SymPy backend is private to this module and loaded
lazily so importing ``jacobian.math`` does not eagerly load packaged backends.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import lcm
from typing import TYPE_CHECKING, Any, cast

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices import _conversions as conversions
from jacobian.math.matrices._operation_models import (
    MAX_CHARACTERISTIC_POLYNOMIAL_ORDER,
    MAX_DETERMINANT_MATRIX_DIMENSION,
    MAX_DETERMINANT_SCALAR_WORK,
    MAX_EXACT_LINEAR_MATRIX_WORK,
    MAX_INPUT_SCALAR_DIGITS,
    MAX_INVERSE_MATRIX_ORDER,
    MAX_INVERSE_OUTPUT_DIGIT_WORK,
    MAX_KRONECKER_PRODUCT_AXIS,
    MAX_MATRIX_PRODUCT_AXIS,
    MAX_MATRIX_PRODUCT_MULTIPLY_ADDS,
    MAX_MATRIX_PRODUCT_OUTPUT_DIGIT_WORK,
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
from jacobian.math.matrices.values import (
    MAX_EXACT_LINEAR_MATRIX_AXIS,
    MAX_MATRIX_DIMENSION,
    MAX_MATRIX_SCALAR_DIGITS,
    IntegerMatrix,
    RationalMatrix,
    SmithNormalForm,
    rational_matrix_from_fractions,
)

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


def _rational_scalars_within_kernel_limit(value: MatrixBase) -> bool:
    return all(
        max(
            _positive_decimal_digits(int(entry.p)),
            _positive_decimal_digits(int(entry.q)),
        )
        <= MAX_INPUT_SCALAR_DIGITS
        for entry in value
    )


def rref(matrix: MatrixBase) -> tuple[MatrixBase, tuple[int, ...]]:
    result = rref_result(
        conversions.rational_matrix_from_sympy(
            _exact_matrix(matrix, maximum_dimension=MAX_EXACT_LINEAR_MATRIX_AXIS)
        )
    )
    return (
        conversions.rational_matrix_to_sympy(result.reduced_matrix),
        result.pivot_columns,
    )


def inverse(matrix: MatrixBase) -> MatrixBase:

    source = _exact_matrix(matrix, maximum_dimension=MAX_INVERSE_MATRIX_ORDER)
    if source.rows != source.cols:
        raise ValueError("inverse requires a square matrix")
    integer_entries = all(entry.is_Integer is True for entry in source)
    input_within_kernel_limit = integer_entries and all(
        _positive_decimal_digits(int(entry)) <= MAX_INPUT_SCALAR_DIGITS
        for entry in source
    )
    if integer_entries and input_within_kernel_limit:
        result = inverse_result(conversions.integer_matrix_from_sympy(source))
        return conversions.rational_matrix_to_sympy(result.inverse)
    if source.rows > MAX_MATRIX_DIMENSION:
        raise ValueError(
            f"matrix dimensions must be between 1 and {MAX_MATRIX_DIMENSION}"
        )
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    try:
        numerator, denominator = DomainMatrix.from_Matrix(source).inv_den()
    except DMNonInvertibleMatrixError as exc:
        raise MatrixSingularError("matrix is singular; inverse does not exist") from exc
    if hasattr(denominator, "x") and hasattr(denominator, "y"):
        import sympy

        denominator = sympy.Integer(denominator.x) + sympy.I * sympy.Integer(
            denominator.y
        )
    return numerator.to_Matrix() / denominator


def trace(matrix: MatrixBase) -> Any:
    import sympy

    source = _exact_matrix(matrix)
    if source.rows != source.cols:
        raise ValueError("trace requires a square matrix")
    return sympy.simplify(source.trace())


def characteristic_polynomial(matrix: MatrixBase, variable: str) -> Any:
    import sympy

    source = _exact_matrix(
        matrix, maximum_dimension=MAX_CHARACTERISTIC_POLYNOMIAL_ORDER
    )
    if source.rows != source.cols:
        raise ValueError("characteristic polynomial requires a square matrix")
    if not all(entry.is_Rational is True for entry in source):
        if source.rows > MAX_MATRIX_DIMENSION:
            raise ValueError(
                f"matrix dimensions must be between 1 and {MAX_MATRIX_DIMENSION}"
            )
        return source.charpoly(variable)
    if not _rational_scalars_within_kernel_limit(source):
        if source.rows > MAX_MATRIX_DIMENSION:
            raise ValueError(
                f"matrix dimensions must be between 1 and {MAX_MATRIX_DIMENSION}"
            )
        return source.charpoly(variable)
    result = characteristic_polynomial_result(
        conversions.rational_matrix_from_sympy(source)
    )
    return sympy.Poly(
        [
            sympy.Rational(coefficient.num, coefficient.den)
            for coefficient in result.coefficients_descending
        ],
        sympy.Symbol(variable),
    )


def determinant(matrix: MatrixBase) -> Any:
    import sympy

    source = _exact_matrix(matrix, maximum_dimension=MAX_DETERMINANT_MATRIX_DIMENSION)
    if source.rows != source.cols:
        raise ValueError("determinant requires a square matrix")
    if not all(entry.is_Rational is True for entry in source):
        if source.rows > MAX_MATRIX_DIMENSION:
            raise ValueError(
                f"matrix dimensions must be between 1 and {MAX_MATRIX_DIMENSION}"
            )
        return source.det(method="bareiss")
    result = determinant_result(conversions.rational_matrix_from_sympy(source))
    value = result.determinant.as_fraction()
    return sympy.Rational(value.numerator, value.denominator)


def rank(matrix: MatrixBase) -> tuple[int, tuple[int, ...]]:
    result = rank_result(
        conversions.rational_matrix_from_sympy(
            _exact_matrix(matrix, maximum_dimension=MAX_EXACT_LINEAR_MATRIX_AXIS)
        )
    )
    return result.rank, result.pivot_columns


def smith_normal_form(matrix: MatrixBase) -> MatrixBase:
    result = smith_normal_form_result(
        conversions.integer_matrix_from_sympy(
            _exact_matrix(matrix, maximum_dimension=MAX_EXACT_LINEAR_MATRIX_AXIS)
        )
    )
    return conversions.integer_matrix_to_sympy(result.normal_form)


def multiply(left: MatrixBase, right: MatrixBase) -> MatrixBase:
    from sympy.matrices.matrixbase import MatrixBase as SymPyMatrix

    if (
        isinstance(left, SymPyMatrix)
        and isinstance(right, SymPyMatrix)
        and all(entry.is_Rational is True for entry in left)
        and all(entry.is_Rational is True for entry in right)
    ):
        left_source = _exact_matrix(left, maximum_dimension=MAX_MATRIX_PRODUCT_AXIS)
        right_source = _exact_matrix(right, maximum_dimension=MAX_MATRIX_PRODUCT_AXIS)
        if not (
            _rational_scalars_within_kernel_limit(left_source)
            and _rational_scalars_within_kernel_limit(right_source)
        ):
            if (
                left_source.rows > MAX_MATRIX_DIMENSION
                or left_source.cols > MAX_MATRIX_DIMENSION
                or right_source.rows > MAX_MATRIX_DIMENSION
                or right_source.cols > MAX_MATRIX_DIMENSION
            ):
                raise ValueError(
                    f"matrix dimensions must be between 1 and {MAX_MATRIX_DIMENSION}"
                )
            return left_source * right_source
        result = product_result(
            conversions.rational_matrix_from_sympy(left_source),
            conversions.rational_matrix_from_sympy(right_source),
        )
        return conversions.rational_matrix_to_sympy(result.product)
    left_fallback = _exact_matrix(left, maximum_dimension=MAX_MATRIX_DIMENSION)
    right_fallback = _exact_matrix(right, maximum_dimension=MAX_MATRIX_DIMENSION)
    return left_fallback * right_fallback


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


def _admit[T](
    check: Callable[..., T], *values: Any, location: tuple[str, ...] = ("matrix",)
) -> T:
    try:
        return check(*values)
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


def _admit_exact_linear_matrix(
    entries: tuple[tuple[CanonicalRational | str, ...], ...],
) -> None:
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    require_matrix_scalar_digits(
        entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
    )
    rows = len(entries)
    columns = len(entries[0])
    if rows > MAX_EXACT_LINEAR_MATRIX_AXIS or columns > MAX_EXACT_LINEAR_MATRIX_AXIS:
        raise _validation_error(
            "budget_exceeded",
            "matrix computation dimensions are limited to "
            f"{MAX_EXACT_LINEAR_MATRIX_AXIS} rows and columns",
        )
    rank_bound = min(rows, columns)
    scalar_digits = max(
        len(component.lstrip("-"))
        for row in entries
        for value in row
        for component in (
            (value,) if isinstance(value, str) else (value.num, value.den)
        )
    )
    work = rows * columns * rank_bound * scalar_digits
    if work > MAX_EXACT_LINEAR_MATRIX_WORK:
        raise _validation_error(
            "budget_exceeded",
            "exact linear algebra exceeds the "
            f"{MAX_EXACT_LINEAR_MATRIX_WORK:,}-unit scalar-work budget",
        )

    row_numerator_bits: list[int] = []
    row_denominator_bits: list[int] = []
    for row in entries:
        if isinstance(row[0], str):
            integer_row = cast(tuple[str, ...], row)
            row_numerator_bits.append(
                max(
                    parse_canonical_integer(value).bit_length() for value in integer_row
                )
                + (columns.bit_length() + 1) // 2
            )
            row_denominator_bits.append(1)
            continue
        rational_row = cast(tuple[CanonicalRational, ...], row)
        fractions = tuple(value.as_fraction() for value in rational_row)
        denominator = lcm(*(value.denominator for value in fractions))
        largest_cleared_numerator = max(
            abs(value.numerator) * (denominator // value.denominator)
            for value in fractions
        )
        row_numerator_bits.append(
            largest_cleared_numerator.bit_length() + (columns.bit_length() + 1) // 2
        )
        row_denominator_bits.append(denominator.bit_length())

    numerator_bits = sum(sorted(row_numerator_bits, reverse=True)[:rank_bound])
    denominator_bits = sum(sorted(row_denominator_bits, reverse=True)[:rank_bound])
    # RREF entries are ratios of minors. After clearing denominators row by
    # row, either canonical component is bounded by one integer-minor bound
    # plus one clearing-denominator bound. Smith factors need only the former,
    # so this shared sum remains conservative for both domains.
    if (
        _bit_bound_decimal_digits(numerator_bits + denominator_bits)
        > MAX_MATRIX_SCALAR_DIGITS
    ):
        raise _validation_error(
            "budget_exceeded",
            "exact linear algebra exceeds the canonical result-height bound",
        )


def _flint_rref(
    matrix: RationalMatrix,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[int, ...]]:
    from jacobian.math.matrices._flint import rational_rref

    entries = tuple(
        tuple(value.as_fraction() for value in row) for row in matrix.entries
    )
    reduced, rank_value = rational_rref(entries)
    pivots = tuple(
        next(column for column, value in enumerate(row) if value)
        for row in reduced[:rank_value]
    )
    return reduced, pivots


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
    if (
        len(matrix.entries) > MAX_MATRIX_DIMENSION
        or len(matrix.entries[0]) > MAX_MATRIX_DIMENSION
    ):
        raise _validation_error(
            "budget_exceeded",
            "integer matrix computation dimensions are limited to "
            f"{MAX_MATRIX_DIMENSION} rows and columns",
        )


def _admit_square_integer(matrix: IntegerMatrix) -> None:
    _admit_integer(matrix)
    rows = len(matrix.entries)
    if rows == 0 or rows != len(matrix.entries[0]):
        raise _validation_error(
            "budget_exceeded", "operation requires a square integer matrix"
        )
    if rows > MAX_MATRIX_DIMENSION:
        raise _validation_error(
            "budget_exceeded",
            "matrix computation dimensions are limited to "
            f"{MAX_MATRIX_DIMENSION} rows and columns",
        )


def _admit_inverse(matrix: IntegerMatrix) -> None:
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    require_matrix_scalar_digits(
        matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
    )
    order = len(matrix.entries)
    if order != len(matrix.entries[0]):
        raise _validation_error(
            "budget_exceeded", "inverse requires a square integer matrix"
        )
    if order > MAX_INVERSE_MATRIX_ORDER:
        raise _validation_error(
            "budget_exceeded",
            f"inverse matrices are limited to order {MAX_INVERSE_MATRIX_ORDER}",
        )
    entries = tuple(tuple(int(value) for value in row) for row in matrix.entries)
    diagonal = tuple(entries[index][index] for index in range(order))
    if all(
        entries[row][column] == 0
        for row in range(order)
        for column in range(order)
        if row != column
    ) and all(value != 0 for value in diagonal):
        output_digit_work = order * order + sum(
            _positive_decimal_digits(value) for value in diagonal
        )
        if output_digit_work > MAX_INVERSE_OUTPUT_DIGIT_WORK:
            raise _validation_error(
                "budget_exceeded",
                "diagonal inverse coefficient work exceeds the exact output budget",
            )
        return
    rank_one_digit_work = _rank_one_inverse_digit_work(entries)
    if (
        rank_one_digit_work is not None
        and rank_one_digit_work <= MAX_INVERSE_OUTPUT_DIGIT_WORK
    ):
        return
    row_squared_norms = tuple(sum(value * value for value in row) for row in entries)
    column_squared_norms = tuple(
        sum(row[column] * row[column] for row in entries) for column in range(order)
    )
    row_determinant_bits, row_cofactor_bits = _hadamard_axis_bits(row_squared_norms)
    column_determinant_bits, column_cofactor_bits = _hadamard_axis_bits(
        column_squared_norms
    )
    component_digits = _bit_bound_decimal_digits(
        max(
            min(row_determinant_bits, column_determinant_bits),
            min(row_cofactor_bits, column_cofactor_bits),
        )
    )
    if order * order * component_digits > MAX_INVERSE_OUTPUT_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "dense inverse coefficient work exceeds the exact output budget",
        )


def _rank_one_inverse_digit_work(
    entries: tuple[tuple[int, ...], ...],
) -> int | None:
    """Bound an inverse of ``I + B`` when ``B`` has rank one.

    For rank-one ``B``, ``B² = trace(B) B``.  If ``1 + trace(B)`` is
    nonzero, Sherman--Morrison gives ``(I + B)⁻¹ = I - B/(1 + trace(B))``.
    Returning the exact component-height bound for this structural case avoids
    charging the generic dense Hadamard estimate to a reconstructible result.
    """

    order = len(entries)
    perturbation = tuple(
        tuple(value - int(row == column) for column, value in enumerate(values))
        for row, values in enumerate(entries)
    )
    pivot_row: tuple[int, ...] | None = None
    pivot_column = 0
    for row in perturbation:
        for column, value in enumerate(row):
            if value != 0:
                pivot_row = row
                pivot_column = column
                break
        if pivot_row is not None:
            break
    if pivot_row is None:
        return order * order
    pivot = pivot_row[pivot_column]
    for row in perturbation:
        if any(
            row[column] * pivot != row[pivot_column] * pivot_row[column]
            for column in range(order)
        ):
            return None
    denominator = 1 + sum(perturbation[index][index] for index in range(order))
    if denominator == 0:
        return None
    component_digits = max(
        max(
            _positive_decimal_digits(
                (denominator if row == column else 0) - perturbation[row][column]
            ),
            _positive_decimal_digits(denominator),
        )
        for row in range(order)
        for column in range(order)
    )
    return order * order + order * order * component_digits


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


def _denominator_digits(denominator: str) -> int:
    return 0 if denominator == "1" else len(denominator)


def _product_cell_digit_bound(
    left_row: tuple[CanonicalRational, ...],
    right_column: tuple[CanonicalRational, ...],
) -> int:
    """Bound one output cell after combining equal-denominator terms."""

    combined_numerators: dict[str, int] = {}
    for left_value, right_value in zip(left_row, right_column, strict=True):
        if left_value.num == "0" or right_value.num == "0":
            continue
        product = left_value.as_fraction() * right_value.as_fraction()
        key = format_canonical_integer(product.denominator)
        combined_numerators[key] = combined_numerators.get(key, 0) + product.numerator
    remaining = tuple(
        (key, numerator)
        for key, numerator in combined_numerators.items()
        if numerator != 0
    )
    if not remaining:
        return 1
    denominator_digits = max(
        1, sum(_denominator_digits(denominator) for denominator, _ in remaining)
    )
    max_numerator_digits = max(
        1, max(_positive_decimal_digits(numerator) for _, numerator in remaining)
    )
    return max(
        max_numerator_digits + denominator_digits + len(str(len(remaining))),
        denominator_digits,
    )


def _admit_product(left: RationalMatrix, right: RationalMatrix) -> None:
    if len(left.entries[0]) != len(right.entries):
        raise _validation_error(
            "budget_exceeded",
            "matrix multiplication requires the left column count to equal the right row count",
        )
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    for label, matrix in (("left", left), ("right", right)):
        require_matrix_scalar_digits(
            matrix.entries,
            maximum=MAX_INPUT_SCALAR_DIGITS,
            label=f"{label} matrix input",
        )
    left_rows = len(left.entries)
    inner_dimension = len(left.entries[0])
    right_columns = len(right.entries[0])
    if (
        left_rows > MAX_MATRIX_PRODUCT_AXIS
        or inner_dimension > MAX_MATRIX_PRODUCT_AXIS
        or right_columns > MAX_MATRIX_PRODUCT_AXIS
        or len(right.entries) > MAX_MATRIX_PRODUCT_AXIS
    ):
        raise _validation_error(
            "budget_exceeded",
            "matrix product axes are limited to "
            f"{MAX_MATRIX_PRODUCT_AXIS} rows and columns",
        )
    if left_rows * inner_dimension * right_columns > MAX_MATRIX_PRODUCT_MULTIPLY_ADDS:
        raise _validation_error(
            "budget_exceeded",
            "matrix product exceeds the exact multiply-add work budget",
        )
    right_columns_entries = tuple(
        tuple(right.entries[row][column] for row in range(inner_dimension))
        for column in range(right_columns)
    )
    output_digit_work = 0
    for left_row in left.entries:
        for right_column in right_columns_entries:
            cell_digits = _product_cell_digit_bound(left_row, right_column)
            if cell_digits > MAX_CANONICAL_RATIONAL_DIGITS:
                raise _validation_error(
                    "budget_exceeded",
                    "matrix product components exceed the canonical digit budget",
                )
            output_digit_work += cell_digits
    if output_digit_work > MAX_MATRIX_PRODUCT_OUTPUT_DIGIT_WORK:
        raise _validation_error(
            "budget_exceeded",
            "matrix product exceeds the exact dense-output digit budget",
        )


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


def _bit_bound_decimal_digits(bits: int) -> int:
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _positive_decimal_digits(value: int) -> int:
    """Count decimal digits exactly without converting the integer to a string."""

    magnitude = abs(value)
    if magnitude <= 9:
        return 1
    digits = _bit_bound_decimal_digits(magnitude.bit_length())
    lower_bound = 10 ** (digits - 1)
    while magnitude < lower_bound:
        digits -= 1
        lower_bound //= 10
    upper_bound = lower_bound * 10
    while magnitude >= upper_bound:
        digits += 1
        lower_bound = upper_bound
        upper_bound *= 10
    return digits


def _hadamard_axis_bits(squared_norms: tuple[int, ...]) -> tuple[int, int]:
    """Return determinant and max-cofactor bit bounds along one Hadamard axis.

    Zero coordinates contribute a zero Euclidean norm, so sparse rows and
    columns do not inflate the product. Cofactors of ``A`` are bounded by the
    product of the remaining axis norms, hence by the product of the
    ``n - 1`` largest norms.
    """

    axis_bits = tuple((norm.bit_length() + 1) // 2 for norm in squared_norms)
    determinant_bits = sum(axis_bits)
    if len(axis_bits) <= 1:
        return determinant_bits, 0
    return determinant_bits, determinant_bits - min(axis_bits)


def _exceeds_canonical_rational_digits(bits: int) -> bool:
    return _bit_bound_decimal_digits(bits) > MAX_CANONICAL_RATIONAL_DIGITS


def _admit_determinant(
    matrix: RationalMatrix,
) -> tuple[tuple[Fraction, ...], ...]:
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
    entries = tuple(
        tuple(value.as_fraction() for value in row) for row in matrix.entries
    )
    order = len(entries)
    input_digits = max(
        max(len(str(abs(value.numerator))), len(str(value.denominator)))
        for row in entries
        for value in row
    )
    scalar_work = order**3 * input_digits
    if scalar_work > MAX_DETERMINANT_SCALAR_WORK:
        raise _validation_error(
            "budget_exceeded",
            "determinant exceeds the "
            f"{MAX_DETERMINANT_SCALAR_WORK:,}-unit scalar-work budget",
        )
    numerator_bits = 0
    denominator_bits = 0
    for row in entries:
        row_denominator = lcm(*(value.denominator for value in row))
        denominator_bits += row_denominator.bit_length()
        squared_norm = sum(
            (value.numerator * (row_denominator // value.denominator)) ** 2
            for value in row
        )
        numerator_bits += (squared_norm.bit_length() + 1) // 2
    if (
        max(
            _bit_bound_decimal_digits(numerator_bits),
            _bit_bound_decimal_digits(denominator_bits),
        )
        > MAX_MATRIX_SCALAR_DIGITS
    ):
        raise _validation_error(
            "budget_exceeded",
            "determinant exceeds the canonical rational result-height bound",
        )
    return entries


def _characteristic_polynomial_component_digit_bound(
    matrix: RationalMatrix,
) -> int:
    """Bound every coefficient from the row-cleared principal-minor envelope.

    Coefficients are sums of principal minors. Clearing row ``i`` by the LCM
    ``d_i`` of its denominators, Hadamard bounds the cleared numerator of any
    minor that includes that row, and the product of the participating ``d_i``
    bounds that minor's denominator. ``2^n`` bounds the number of minors of
    any fixed order. Unreduced coefficients written over the product of all
    row denominators therefore fit in ``2^n`` times the full-order Hadamard
    bound times that product.
    """
    fractions = tuple(
        tuple(value.as_fraction() for value in row) for row in matrix.entries
    )
    order = len(fractions)
    over_budget = MAX_CANONICAL_RATIONAL_DIGITS + 1
    numerator_bits = 0
    denominator_bits = 0
    is_upper_triangular = all(
        fractions[row][column] == 0 for row in range(order) for column in range(row)
    )
    is_lower_triangular = all(
        fractions[row][column] == 0
        for row in range(order)
        for column in range(row + 1, order)
    )
    triangular = is_upper_triangular or is_lower_triangular
    bound_rows = (
        tuple((fractions[index][index],) for index in range(order))
        if triangular
        else fractions
    )
    seen_rows: set[tuple[Fraction, ...]] = set()
    for row in bound_rows:
        row_denominator = 1
        if not triangular and row in seen_rows:
            continue
        if not triangular:
            seen_rows.add(row)
        for value in row:
            row_denominator = lcm(row_denominator, value.denominator)
            if _exceeds_canonical_rational_digits(
                denominator_bits + row_denominator.bit_length()
            ):
                return over_budget
        denominator_bits += row_denominator.bit_length()
        squared_norm = sum(
            (value.numerator * (row_denominator // value.denominator)) ** 2
            for value in row
        )
        numerator_bits += (squared_norm.bit_length() + 1) // 2
        if _exceeds_canonical_rational_digits(
            numerator_bits + denominator_bits + order
        ):
            return over_budget
    return max(
        _bit_bound_decimal_digits(numerator_bits + denominator_bits + order),
        _bit_bound_decimal_digits(max(1, denominator_bits)),
    )


def _admit_characteristic_polynomial(matrix: RationalMatrix) -> None:
    from jacobian.math.matrices.values import require_matrix_scalar_digits

    require_matrix_scalar_digits(
        matrix.entries, maximum=MAX_INPUT_SCALAR_DIGITS, label="matrix input"
    )
    order = len(matrix.entries)
    if order != len(matrix.entries[0]):
        raise _validation_error(
            "budget_exceeded", "characteristic polynomial requires a square matrix"
        )
    if order > MAX_CHARACTERISTIC_POLYNOMIAL_ORDER:
        raise _validation_error(
            "budget_exceeded",
            "characteristic-polynomial matrices are limited to order "
            f"{MAX_CHARACTERISTIC_POLYNOMIAL_ORDER}",
        )
    if (
        _characteristic_polynomial_component_digit_bound(matrix)
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        raise _validation_error(
            "budget_exceeded",
            "characteristic-polynomial coefficients exceed the canonical digit budget",
        )


def _admit_linear_solve(
    matrix: RationalMatrix, rhs: tuple[CanonicalRational, ...]
) -> None:
    _require_square_system_admission(matrix, rhs)


def determinant_result(matrix: RationalMatrix) -> MatrixDeterminantResult:
    entries = _admit(_admit_determinant, matrix)
    from jacobian.math.matrices._flint import rational_determinant

    value = rational_determinant(entries)
    return MatrixDeterminantResult(determinant=CanonicalRational.from_fraction(value))


def rank_result(matrix: RationalMatrix) -> MatrixRankResult:
    _admit(_admit_exact_linear_matrix, matrix.entries)
    _, pivot_columns = _flint_rref(matrix)
    return MatrixRankResult._from_kernel(
        matrix=matrix,
        rank=len(pivot_columns),
        pivot_columns=tuple(int(column) for column in pivot_columns),
    )


def rref_result(matrix: RationalMatrix) -> RrefResult:
    _admit(_admit_exact_linear_matrix, matrix.entries)
    reduced, pivots = _flint_rref(matrix)
    columns = len(reduced[0])
    pivot_columns = tuple(int(column) for column in pivots)
    return RrefResult._from_kernel(
        matrix=matrix,
        reduced_matrix=rational_matrix_from_fractions(reduced),
        rank=len(pivot_columns),
        pivot_columns=pivot_columns,
        free_columns=tuple(
            column for column in range(columns) if column not in pivot_columns
        ),
    )


def nullspace_result(matrix: RationalMatrix) -> NullspaceResult:
    _admit(_admit_exact_linear_matrix, matrix.entries)
    reduced, pivots = _flint_rref(matrix)
    pivot_columns = tuple(int(column) for column in pivots)
    free_columns = tuple(
        column for column in range(len(reduced[0])) if column not in pivot_columns
    )
    pivot_row_by_column = {
        pivot_column: row for row, pivot_column in enumerate(pivot_columns)
    }
    basis: list[tuple[CanonicalRational, ...]] = []
    for free_column in free_columns:
        vector = [Fraction(0)] * len(reduced[0])
        vector[free_column] = Fraction(1)
        for pivot_column, row in pivot_row_by_column.items():
            vector[pivot_column] = -reduced[row][free_column]
        basis.append(tuple(CanonicalRational.from_fraction(value) for value in vector))
    return NullspaceResult._from_kernel(
        matrix=matrix,
        ambient_dimension=len(reduced[0]),
        rank=len(pivot_columns),
        nullity=len(basis),
        basis_vectors=tuple(basis),
        free_columns=free_columns,
    )


def characteristic_polynomial_result(
    matrix: RationalMatrix,
) -> CharacteristicPolynomialResult:
    _admit(_admit_characteristic_polynomial, matrix)
    from flint import fmpq, fmpq_mat

    order = len(matrix.entries)
    entries = [
        fmpq(*value.as_integer_ratio()) for row in matrix.entries for value in row
    ]
    polynomial = fmpq_mat(order, order, entries).charpoly()
    return CharacteristicPolynomialResult(
        degree=order,
        coefficients_descending=tuple(
            CanonicalRational.from_integer_ratio(
                int(polynomial[index].p), int(polynomial[index].q)
            )
            for index in range(order, -1, -1)
        ),
    )


def smith_normal_form_result(matrix: IntegerMatrix) -> SmithNormalForm:
    _admit(_admit_exact_linear_matrix, matrix.entries)
    from jacobian.math.matrices._flint import integer_smith_normal_form

    raw = integer_smith_normal_form(
        tuple(
            tuple(parse_canonical_integer(value) for value in row)
            for row in matrix.entries
        )
    )
    diagonal = tuple(raw[index][index] for index in range(min(len(raw), len(raw[0]))))
    rank_value = next(
        (index for index, value in enumerate(diagonal) if value == 0), len(diagonal)
    )
    factors = tuple(abs(value) for value in diagonal[:rank_value])
    normal_form = IntegerMatrix(
        entries=tuple(
            tuple(
                format_canonical_integer(factors[row])
                if row == column and row < rank_value
                else "0"
                for column in range(len(raw[0]))
            )
            for row in range(len(raw))
        )
    )
    return SmithNormalForm(
        normal_form=normal_form,
        rank=rank_value,
        invariant_factors=tuple(format_canonical_integer(value) for value in factors),
    )


def inverse_result(matrix: IntegerMatrix) -> MatrixInverseResult:
    _admit(_admit_inverse, matrix)
    from flint import fmpq_mat

    order = len(matrix.entries)
    source = fmpq_mat(
        order,
        order,
        [int(value) for row in matrix.entries for value in row],
    )
    try:
        value = source.inv()
    except ZeroDivisionError as exc:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="matrix.singular_matrix",
            message="matrix is singular; inverse does not exist",
        ) from exc
    return MatrixInverseResult(
        inverse=RationalMatrix(
            entries=tuple(
                tuple(
                    CanonicalRational.from_integer_ratio(
                        int(value[row, column].p), int(value[row, column].q)
                    )
                    for column in range(order)
                )
                for row in range(order)
            )
        )
    )


def trace_result(matrix: IntegerMatrix) -> MatrixTraceResult:
    _admit(_admit_square_integer, matrix)
    return MatrixTraceResult(
        trace=format_canonical_integer(
            trace(conversions.integer_matrix_to_sympy(matrix))
        )
    )


def product_result(left: RationalMatrix, right: RationalMatrix) -> MatrixProductResult:
    _admit(_admit_product, left, right, location=("left", "right"))
    from jacobian.math.matrices._flint import rational_matrix_product

    left_rows = len(left.entries)
    inner_dimension = len(left.entries[0])
    right_columns = len(right.entries[0])
    product = rational_matrix_product(
        tuple(tuple(value.as_fraction() for value in row) for row in left.entries),
        tuple(tuple(value.as_fraction() for value in row) for row in right.entries),
    )
    return MatrixProductResult(
        product=RationalMatrix(
            entries=tuple(
                tuple(CanonicalRational.from_fraction(value) for value in row)
                for row in product
            )
        ),
        left_rows=left_rows,
        inner_dimension=inner_dimension,
        right_columns=right_columns,
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
