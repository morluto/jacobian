"""Exact integral congruence changes of quadratic-form coordinates."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
)
from jacobian.math.number_theory.quadratic_forms.integral._models import (
    MAX_INTEGRAL_QUADRATIC_FORM_TERMS,
    IntegralQuadraticCrossTerm,
    IntegralQuadraticForm,
)
from jacobian.math.number_theory.quadratic_forms.integral.unimodular._models import (
    MAX_UNIMODULAR_CHANGE_AXIS,
    UnimodularChangeRequest,
    UnimodularChangeResult,
)

MAX_CHANGE_MATRIX_ENTRY_DIGITS = 128
MAX_CHANGE_OUTPUT_INTEGER_DIGITS = 8_192
MAX_CHANGE_RESULT_DECIMAL_DIGITS = 9_000_000
MAX_CHANGE_WORK = 500_000


def _failure(reason: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("matrix",),
        code=f"quadratic_form.unimodular.{reason}",
        message=message,
    )


def _resource_failure(reason: str, message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("matrix",),
        code=f"quadratic_form.unimodular.{reason}",
        message=message,
    )


def _digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _decimal_digit_upper_bound(bit_length: int) -> int:
    """Return a safe decimal digit ceiling without expanding a large power."""

    if bit_length <= 0:
        return 1
    return (bit_length * 30_103 + 99_999) // 100_000


def _determinant(matrix: list[list[int]]) -> int:
    """Fraction-free Bareiss determinant, including det of the empty matrix."""

    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for pivot_index in range(n - 1):
        pivot_row = next(
            (row for row in range(pivot_index, n) if work[row][pivot_index]), None
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, n):
            for column in range(pivot_index + 1, n):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                work[row][column] = numerator // previous
            work[row][pivot_index] = 0
        previous = pivot
    return sign * work[-1][-1]


def _inverse_unimodular(matrix: list[list[int]]) -> list[list[int]]:
    """Exact Gauss-Jordan inverse; the unimodular input makes it integral."""

    n = len(matrix)
    augmented = [
        [Fraction(value) for value in row]
        + [Fraction(int(row_index == column)) for column in range(n)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = next((row for row in range(column, n) if augmented[row][column]), None)
        if pivot is None:
            raise _failure("internal_singularity", "unimodular matrix became singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    result: list[list[int]] = []
    for augmented_row in augmented:
        inverse_row: list[int] = []
        for value in augmented_row[n:]:
            if value.denominator != 1:
                raise _failure(
                    "internal_nonintegral_inverse",
                    "determinant-one integer matrix must have an integer inverse",
                )
            inverse_row.append(value.numerator)
        result.append(inverse_row)
    return result


def _require_canonical_form(form: IntegralQuadraticForm) -> int:
    n = len(form.axis)
    if n > MAX_UNIMODULAR_CHANGE_AXIS:
        raise _failure("axis_bound", "unimodular change axis exceeds 32")
    if (
        form.domain != "ZZ"
        or any(not isinstance(label, str) for label in form.axis)
        or len(set(form.axis)) != n
        or len(form.diagonal_coefficients) != n
        or n + len(form.cross_terms) > MAX_INTEGRAL_QUADRATIC_FORM_TERMS
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or _digits(value) > MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
            for value in form.diagonal_coefficients
        )
    ):
        raise _failure("form_shape", "form must be a canonical bounded ZZ polynomial")
    positions: list[tuple[int, int]] = []
    for term in form.cross_terms:
        if not isinstance(term, IntegralQuadraticCrossTerm):
            raise _failure("cross_term_type", "cross terms must be canonical values")
        if (
            not isinstance(term.left, int)
            or isinstance(term.left, bool)
            or not isinstance(term.right, int)
            or isinstance(term.right, bool)
            or term.left < 0
            or term.left >= term.right
            or term.right >= n
            or not isinstance(term.coefficient, int)
            or isinstance(term.coefficient, bool)
            or term.coefficient == 0
            or _digits(term.coefficient) > MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
        ):
            raise _failure("cross_term_shape", "form cross terms must be canonical")
        positions.append((term.left, term.right))
    if positions != sorted(set(positions)):
        raise _failure(
            "cross_term_order", "form cross terms must be ordered and unique"
        )
    return n


def _preflight(request: UnimodularChangeRequest) -> list[list[int]]:
    if not isinstance(request.form, IntegralQuadraticForm):
        raise _failure("form_type", "expected a canonical integral quadratic form")
    if not isinstance(request.matrix, IntegerMatrix):
        raise _failure("matrix_type", "expected a canonical integer matrix")
    n = _require_canonical_form(request.form)
    if (
        request.matrix.row_count != n
        or request.matrix.column_count != n
        or len(request.matrix.entries) != n
        or len(request.target_axis) != n
        or any(not isinstance(label, str) for label in request.target_axis)
        or len(set(request.target_axis)) != n
        or len(request.form.diagonal_coefficients) != n
    ):
        raise _failure("shape", "form, matrix, and axes must share one dimension")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for row in request.matrix.entries
        for value in row
    ) or any(len(row) != n for row in request.matrix.entries):
        raise _failure(
            "matrix_entries", "matrix entries must form a square integer matrix"
        )
    matrix = [[int(value) for value in row] for row in request.matrix.entries]
    entry_digits = max((_digits(value) for row in matrix for value in row), default=1)
    if entry_digits > MAX_CHANGE_MATRIX_ENTRY_DIGITS:
        raise _resource_failure(
            "matrix_entry_bound",
            f"change matrix entries are limited to {MAX_CHANGE_MATRIX_ENTRY_DIGITS} digits",
        )

    # Every inverse entry is an (n-1)-minor. Hadamard's inequality and the
    # larger n*max-entry row bound give a cheap conservative decimal envelope.
    max_entry = max((abs(value) for row in matrix for value in row), default=0)
    factor_bits = max(1, n * max_entry).bit_length()
    inverse_digits = _decimal_digit_upper_bound(factor_bits * max(0, n - 1))
    if inverse_digits > MAX_CHANGE_OUTPUT_INTEGER_DIGITS:
        raise _resource_failure(
            "inverse_growth_bound",
            "the conservative exact inverse bound exceeds the admitted output digits",
        )

    coefficient_max = max(
        [abs(value) for value in request.form.diagonal_coefficients]
        + [abs(term.coefficient) for term in request.form.cross_terms]
        + [0]
    )
    polar_max = max(2 * coefficient_max, coefficient_max)
    column_sums = [
        sum(abs(matrix[row][column]) for row in range(n)) for column in range(n)
    ]
    target_bound = max(
        (
            polar_max * column_sums[left] * column_sums[right]
            for left in range(n)
            for right in range(n)
        ),
        default=0,
    )
    target_digits = _digits(target_bound)
    if target_digits > 256:
        raise _resource_failure(
            "coefficient_growth_bound",
            "transformed integral coefficients exceed the 256-digit coefficient bound",
        )

    source_digits = sum(
        _digits(value) for value in request.form.diagonal_coefficients
    ) + sum(_digits(term.coefficient) for term in request.form.cross_terms)
    matrix_digits = sum(_digits(value) for row in matrix for value in row)
    target_support = n + n * (n - 1) // 2
    result_digits = (
        source_digits
        + matrix_digits
        + n * n * inverse_digits
        + target_support * target_digits
    )
    if result_digits > MAX_CHANGE_RESULT_DECIMAL_DIGITS:
        raise _resource_failure(
            "result_growth_bound",
            "source, transport maps, and transformed form exceed the aggregate result bound",
        )
    estimated_work = 5 * n**3 + n * n
    if estimated_work > MAX_CHANGE_WORK:
        raise _resource_failure(
            "work_bound", "unimodular change exceeds the admitted work bound"
        )
    return matrix


def unimodular_change(
    request: UnimodularChangeRequest,
) -> UnimodularChangeResult:
    """Return ``Q(M y)`` after proving the supplied integer matrix is unimodular."""

    if not isinstance(request, UnimodularChangeRequest):
        raise _failure("request_type", "expected a typed unimodular change request")
    matrix = _preflight(request)
    determinant = _determinant(matrix)
    if determinant not in (-1, 1):
        raise _failure(
            "determinant_not_unit",
            "unimodular coordinate changes require determinant 1 or -1",
        )

    inverse = _inverse_unimodular(matrix)
    n = len(matrix)
    polar = [[0] * n for _ in range(n)]
    for index, coefficient in enumerate(request.form.diagonal_coefficients):
        polar[index][index] = 2 * coefficient
    for term in request.form.cross_terms:
        polar[term.left][term.right] = term.coefficient
        polar[term.right][term.left] = term.coefficient

    # Compute M^T B M with exact integers. B is the polar matrix, so its
    # transformed diagonal is twice the transformed polynomial's diagonal.
    right_product = [
        [
            sum(polar[row][inner] * matrix[inner][column] for inner in range(n))
            for column in range(n)
        ]
        for row in range(n)
    ]
    transformed_polar = [
        [
            sum(matrix[inner][row] * right_product[inner][column] for inner in range(n))
            for column in range(n)
        ]
        for row in range(n)
    ]
    diagonal: list[int] = []
    for index in range(n):
        value = transformed_polar[index][index]
        if value % 2:
            raise _failure(
                "internal_polar_parity", "transformed polar diagonal must be even"
            )
        diagonal.append(value // 2)
    cross_terms = tuple(
        IntegralQuadraticCrossTerm(left=left, right=right, coefficient=value)
        for left in range(n)
        for right in range(left + 1, n)
        if (value := transformed_polar[left][right]) != 0
    )
    target = IntegralQuadraticForm(
        axis=request.target_axis,
        diagonal_coefficients=tuple(diagonal),
        cross_terms=cross_terms,
    )
    return UnimodularChangeResult(
        source=request.form,
        matrix=IntegerMatrix(
            row_count=n,
            column_count=n,
            entries=tuple(tuple(row) for row in matrix),
        ),
        inverse=IntegerMatrix(
            row_count=n,
            column_count=n,
            entries=tuple(tuple(row) for row in inverse),
        ),
        target=target,
    )


__all__ = ["unimodular_change"]
