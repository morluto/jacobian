"""Admitted fraction-free determinant kernel for rational quadratic forms."""

from __future__ import annotations

from fractions import Fraction
from math import gcd, lcm

from jacobian._execution import request_checkpoint
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general.determinant_discriminant._models import (
    MAX_POLAR_DETERMINANT_AXIS,
    MAX_POLAR_DETERMINANT_INTERMEDIATE_DIGITS,
    MAX_POLAR_DETERMINANT_MATRIX_ENTRIES,
    MAX_POLAR_DETERMINANT_OUTPUT_DIGITS,
    MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS,
    MAX_POLAR_DETERMINANT_SUPPORT_TERMS,
    MAX_POLAR_DETERMINANT_WORK,
    DeterminantDiscriminantRequest,
    DeterminantDiscriminantResult,
)


def _digits(value: str | int) -> int:
    return (
        len(value.lstrip("-")) if isinstance(value, str) else decimal_digit_width(value)
    )


def _ceil_log10(value: int) -> int:
    return 0 if value <= 1 else len(str(value - 1))


def _preflight(
    request: DeterminantDiscriminantRequest,
) -> None:
    """Admit dimension, dense work, output height, and row denominators.

    Clearing each polar-Gram row by the product of its stored denominators
    gives an integer matrix. Every Bareiss entry is then a minor of that
    matrix and Hadamard bounds its height by the sum of the cleared row-norm
    digit bounds. Each unreduced Bareiss numerator is a difference of two
    products of such minors, so twice that bound plus two digits covers every
    exact intermediate before the kernel starts.
    """

    if not isinstance(request, DeterminantDiscriminantRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="quadratic_form.determinant_request_type",
            message="request must be a DeterminantDiscriminantRequest",
        )
    request_checkpoint("before quadratic-form determinant admission")
    form = request.form
    n = len(form.axis)
    support = n + len(form.cross_terms)
    if n > MAX_POLAR_DETERMINANT_AXIS:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.determinant_axis_bound",
            message=(
                "polar-Gram determinant dimension exceeds "
                f"{MAX_POLAR_DETERMINANT_AXIS} coordinates"
            ),
        )
    if support > MAX_POLAR_DETERMINANT_SUPPORT_TERMS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.determinant_support_bound",
            message=(
                "polar-Gram determinant source exceeds "
                f"{MAX_POLAR_DETERMINANT_SUPPORT_TERMS} stored terms"
            ),
        )
    if n * n > MAX_POLAR_DETERMINANT_MATRIX_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.determinant_matrix_bound",
            message=(
                "polar-Gram determinant matrix exceeds "
                f"{MAX_POLAR_DETERMINANT_MATRIX_ENTRIES} entries"
            ),
        )

    maximum_coefficient_digits = 1
    retained_digits = sum(
        _digits(coefficient.num) + _digits(coefficient.den)
        for coefficient in form.diagonal_coefficients
    )
    for term in form.cross_terms:
        maximum_coefficient_digits = max(
            maximum_coefficient_digits,
            _digits(term.coefficient.num),
            _digits(term.coefficient.den),
        )
        retained_digits += _digits(term.coefficient.num) + _digits(term.coefficient.den)
    for coefficient in form.diagonal_coefficients:
        maximum_coefficient_digits = max(
            maximum_coefficient_digits,
            _digits(coefficient.num),
            _digits(coefficient.den),
        )

    # Keep source retention itself inside this operation's output envelope.
    # Axis labels are bounded by the shared 64-character OpaqueLabel contract.
    retained_digits += 64 * n
    if retained_digits > MAX_POLAR_DETERMINANT_RETAINED_SOURCE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.determinant_source_output_bound",
            message="retained quadratic-form source exceeds the result envelope",
        )

    work = n**3 * maximum_coefficient_digits
    if work > MAX_POLAR_DETERMINANT_WORK:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.determinant_work_bound",
            message="polar-Gram determinant exceeds the admitted Bareiss work",
        )

    row_entry_meta: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for index, coefficient in enumerate(form.diagonal_coefficients):
        doubled_numerator = 2 * coefficient.num
        common_factor = gcd(abs(doubled_numerator), coefficient.den)
        row_entry_meta[index].append(
            (
                _digits(doubled_numerator // common_factor),
                coefficient.den // common_factor,
            )
        )
    for term in form.cross_terms:
        entry = (_digits(term.coefficient.num), term.coefficient.den)
        for row in (term.left, term.right):
            row_entry_meta[row].append(entry)

    row_norm_digits: list[int] = []
    row_denominator_digits: list[int] = []
    for entries in row_entry_meta:
        common_denominator = lcm(*(denominator for _, denominator in entries))
        denominator_digits = _digits(common_denominator)
        row_denominator_digits.append(denominator_digits)
        cleared_entry_digits = max(
            numerator_digits + _digits(common_denominator // denominator)
            for numerator_digits, denominator in entries
        )
        # The Euclidean norm of a row is below n times its largest entry.
        row_norm_digits.append(cleared_entry_digits + _ceil_log10(n) + 1)

    determinant_numerator_digits = sum(row_norm_digits)
    determinant_denominator_digits = sum(row_denominator_digits)
    intermediate_digits = 2 * determinant_numerator_digits + 2
    if max(determinant_numerator_digits, determinant_denominator_digits) > (
        MAX_POLAR_DETERMINANT_OUTPUT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.determinant_result_height_bound",
            message="determinant or discriminant exceeds the exact output digit bound",
        )
    if intermediate_digits > MAX_POLAR_DETERMINANT_INTERMEDIATE_DIGITS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.determinant_intermediate_height_bound",
            message="Bareiss determinant intermediates exceed the admitted digit bound",
        )
    return None


def _polar_gram_rows(
    request: DeterminantDiscriminantRequest,
) -> tuple[tuple[Fraction, ...], ...]:
    form = request.form
    n = len(form.axis)
    rows = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for index, coefficient in enumerate(form.diagonal_coefficients):
        rows[index][index] = 2 * coefficient.as_fraction()
    for term in form.cross_terms:
        value = term.coefficient.as_fraction()
        rows[term.left][term.right] = value
        rows[term.right][term.left] = value
    return tuple(tuple(row) for row in rows)


def _bareiss_determinant(matrix: list[list[int]]) -> int:
    """Compute a determinant exactly by fraction-free Bareiss elimination."""

    n = len(matrix)
    if n == 0:
        return 1
    sign = 1
    previous_pivot = 1
    request_checkpoint("before Bareiss determinant elimination")
    for pivot_index in range(n - 1):
        request_checkpoint("during Bareiss determinant elimination")
        pivot_row = next(
            (row for row in range(pivot_index, n) if matrix[row][pivot_index] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            matrix[pivot_index], matrix[pivot_row] = (
                matrix[pivot_row],
                matrix[pivot_index],
            )
            sign = -sign
        pivot = matrix[pivot_index][pivot_index]
        for row in range(pivot_index + 1, n):
            entry = matrix[row][pivot_index]
            for column in range(pivot_index + 1, n):
                numerator = (
                    matrix[row][column] * pivot - entry * matrix[pivot_index][column]
                )
                quotient, remainder = divmod(numerator, previous_pivot)
                if remainder:
                    raise ArithmeticError("Bareiss division was not exact")
                matrix[row][column] = quotient
            matrix[row][pivot_index] = 0
        previous_pivot = pivot
    return sign * matrix[-1][-1]


def polar_gram_determinant_discriminant(
    request: DeterminantDiscriminantRequest,
) -> DeterminantDiscriminantResult:
    """Return ``det(G)`` and ``(-1)^(n(n-1)/2) det(G)`` exactly."""

    _preflight(request)
    request_checkpoint("before polar Gram matrix construction")
    rows = _polar_gram_rows(request)
    if not rows:
        return DeterminantDiscriminantResult._from_kernel(
            request, determinant=1, denominator=1
        )
    row_denominators = tuple(lcm(*(value.denominator for value in row)) for row in rows)
    integer_matrix = [
        [
            value.numerator * (row_denominators[index] // value.denominator)
            for value in row
        ]
        for index, row in enumerate(rows)
    ]
    integer_determinant = _bareiss_determinant(integer_matrix)
    request_checkpoint("before determinant result construction")
    denominator = 1
    for row_denominator in row_denominators:
        denominator *= row_denominator
    result = DeterminantDiscriminantResult._from_kernel(
        request,
        determinant=integer_determinant,
        denominator=denominator,
    )
    return result


__all__ = ["polar_gram_determinant_discriminant"]
