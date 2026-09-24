"""Exact arithmetic in the bounded rational cyclotomic coefficient fields.

This is an internal kernel shared by character-valued modular-form operations.
The public value remains ``RationalCyclotomicElement`` from cyclic-linear.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import gcd

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE,
    MAX_MODULAR_FORM_COEFFICIENT_FIELD_ORDER,
)

_MAX_WORK = 1_000_000


def _fail_domain(code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("elements",), code=code, message=message
    )


def _fail_resource(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("elements",), code=code, message=message
    )


def _decimal_digits(value: int) -> int:
    """Count decimal digits without converting untrusted-size integers to text."""
    magnitude = abs(value)
    return 1 if magnitude == 0 else len(str(magnitude))


def _validate_element(
    value: RationalCyclotomicElement,
) -> tuple[RationalCyclotomicField, tuple[Fraction, ...], int]:
    if type(value) is not RationalCyclotomicElement:
        _fail_domain(
            "modular_form.cyclotomic_element_type",
            "cyclotomic arithmetic requires canonical RationalCyclotomicElement values",
        )
    field = getattr(value, "field", None)
    if type(field) is not RationalCyclotomicField:
        _fail_domain(
            "modular_form.cyclotomic_field_type",
            "cyclotomic elements require a canonical RationalCyclotomicField parent",
        )
    if (
        type(getattr(field, "order", None)) is not int
        or not 1
        <= getattr(field, "order", 0)
        <= MAX_MODULAR_FORM_COEFFICIENT_FIELD_ORDER
        or getattr(field, "domain", None) != "QQ_CYCLOTOMIC"
        or getattr(field, "generator", None) != "CLASS_OF_X"
    ):
        _fail_resource(
            "modular_form.cyclotomic_field_bound",
            "cyclotomic arithmetic exceeds the admitted coefficient-field order",
        )
    degree = sum(gcd(index, field.order) == 1 for index in range(1, field.order + 1))
    if not 1 <= degree <= MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE:
        _fail_resource(
            "modular_form.cyclotomic_field_bound",
            "cyclotomic arithmetic exceeds the admitted coefficient-field degree",
        )
    coefficients = getattr(value, "coefficients_ascending", None)
    if type(coefficients) is not tuple or len(coefficients) != degree:
        _fail_domain(
            "modular_form.cyclotomic_coordinate_count",
            "a cyclotomic element needs exactly phi(order) canonical coordinates",
        )
    fractions: list[Fraction] = []
    digits = 1
    limit = 10**MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    for coordinate in coefficients:
        if type(coordinate) is not CanonicalRational:
            _fail_domain(
                "modular_form.cyclotomic_coordinate_type",
                "cyclotomic coordinates must be canonical exact rationals",
            )
        numerator = getattr(coordinate, "num", None)
        denominator = getattr(coordinate, "den", None)
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
            or abs(numerator) >= limit
            or denominator >= limit
            or gcd(abs(numerator), denominator) != 1
        ):
            _fail_domain(
                "modular_form.cyclotomic_coordinate_canonicality",
                "cyclotomic coordinate is not a reduced rational within the height bound",
            )
        digits = max(digits, _decimal_digits(numerator), _decimal_digits(denominator))
        fractions.append(Fraction(numerator, denominator))
    return field, tuple(fractions), digits


def _canonical(
    field: RationalCyclotomicField, values: tuple[Fraction, ...]
) -> RationalCyclotomicElement:
    if len(values) != field.degree:
        raise ArithmeticError(
            "cyclotomic kernel returned a noncanonical coordinate count"
        )
    if any(
        max(len(str(abs(value.numerator))), len(str(value.denominator)))
        > MAX_CYCLIC_FIELD_ELEMENT_DIGITS
        for value in values
    ):
        _fail_resource(
            "modular_form.cyclotomic_height_bound",
            "cyclotomic arithmetic result exceeds the exact coordinate height bound",
        )
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(value) for value in values
        ),
    )


def _admit(
    left: RationalCyclotomicElement,
    right: RationalCyclotomicElement | None,
    *,
    cost: int,
    inverse_operand: RationalCyclotomicElement | None = None,
    include_left_in_inverse_bound: bool = False,
) -> tuple[RationalCyclotomicField, tuple[Fraction, ...], tuple[Fraction, ...] | None]:
    field, left_values, left_digits = _validate_element(left)
    if right is not None:
        right_field, right_values, right_digits = _validate_element(right)
    else:
        right_field, right_values, right_digits = None, None, 0
    if right_field is not None and field != right_field:
        _fail_domain(
            "modular_form.cyclotomic_field_mismatch",
            "cyclotomic arithmetic requires elements of the identical explicit field",
        )
    degree = field.degree
    if cost > _MAX_WORK:
        _fail_resource(
            "modular_form.cyclotomic_work_bound",
            "cyclotomic arithmetic exceeds the exact work bound",
        )
    max_digits = max(left_digits, right_digits)
    if inverse_operand is None:
        predicted_digits = max_digits * (2 * degree + 2) + len(str(degree)) + 2
    else:
        # Clear all coefficient denominators before forming the multiplication
        # matrix. For division, include the left operand's denominators in the
        # same common denominator so this bound covers the quotient solve too.
        denominator_factor = 2 if include_left_in_inverse_bound else 1
        cyclotomic_coefficient_bound = 1 + degree * (1 << degree)
        companion_power_bound = cyclotomic_coefficient_bound ** (2 * degree - 2)
        reduction_digits = len(str(companion_power_bound))
        entry_digits = (
            denominator_factor * degree * max_digits
            + len(str(degree))
            + reduction_digits
            + 2
        )
        # The companion matrix for multiplication by zeta has infinity norm
        # at most 1 + degree*2**degree, since Phi_n coefficients are bounded by
        # the elementary-symmetric estimate 2**degree. Its (2*degree-2) power
        # bounds every reduced x^k coefficient used in the multiplication
        # matrix, whose raw products have degree at most 2*degree-2.
        # Hadamard's bound then bounds every minor, hence every
        # normalized Gauss-Jordan intermediate and Cramer numerator/denominator.
        predicted_digits = degree * entry_digits + degree * len(str(degree)) + 2
    if predicted_digits > MAX_CYCLIC_FIELD_ELEMENT_DIGITS:
        _fail_resource(
            "modular_form.cyclotomic_height_admission",
            "cyclotomic input height exceeds the conservative arithmetic admission bound",
        )
    return field, left_values, right_values


@lru_cache(maxsize=128)
def _cyclotomic_polynomial(order: int) -> tuple[int, ...]:
    """Return ascending integer coefficients of Phi_order by exact division."""
    polynomial = [0] * (order + 1)
    polynomial[0], polynomial[order] = -1, 1
    for divisor in range(1, order):
        if order % divisor:
            continue
        divisor_poly = _cyclotomic_polynomial(divisor)
        quotient = [0] * (len(polynomial) - len(divisor_poly) + 1)
        remainder = polynomial[:]
        for shift in range(len(quotient) - 1, -1, -1):
            coefficient = remainder[shift + len(divisor_poly) - 1]
            quotient[shift] = coefficient
            if coefficient:
                for index, part in enumerate(divisor_poly):
                    remainder[shift + index] -= coefficient * part
        if any(remainder):
            raise ArithmeticError("cyclotomic polynomial division was not exact")
        polynomial = quotient
    return tuple(polynomial)


def _reduce(
    values: list[Fraction], field: RationalCyclotomicField
) -> tuple[Fraction, ...]:
    relation = _cyclotomic_polynomial(field.order)
    degree = len(relation) - 1
    while len(values) > degree:
        coefficient = values.pop()
        if coefficient:
            shift = len(values) - degree
            for index in range(degree):
                values[shift + index] -= coefficient * relation[index]
    return tuple(values + [Fraction(0)] * (degree - len(values)))


def add(
    left: RationalCyclotomicElement, right: RationalCyclotomicElement
) -> RationalCyclotomicElement:
    field, a, b = _admit(left, right, cost=MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE)
    assert b is not None
    return _canonical(field, tuple(x + y for x, y in zip(a, b, strict=True)))


def subtract(
    left: RationalCyclotomicElement, right: RationalCyclotomicElement
) -> RationalCyclotomicElement:
    field, a, b = _admit(left, right, cost=MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE)
    assert b is not None
    return _canonical(field, tuple(x - y for x, y in zip(a, b, strict=True)))


def multiply(
    left: RationalCyclotomicElement, right: RationalCyclotomicElement
) -> RationalCyclotomicElement:
    field, a, b = _admit(
        left,
        right,
        cost=MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE**2 * 3,
    )
    assert b is not None
    return _multiply_coordinates(field, a, b)


def _multiply_coordinates(
    field: RationalCyclotomicField,
    a: tuple[Fraction, ...],
    b: tuple[Fraction, ...],
) -> RationalCyclotomicElement:
    degree = field.degree
    product = [Fraction(0)] * (2 * degree - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] += x * y
    return _canonical(field, _reduce(product, field))


def inverse(value: RationalCyclotomicElement) -> RationalCyclotomicElement:
    field, a, _ = _admit(
        value,
        None,
        cost=MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE**3 * 4,
        inverse_operand=value,
    )
    return _inverse_coordinates(field, a)


def _inverse_coordinates(
    field: RationalCyclotomicField, a: tuple[Fraction, ...]
) -> RationalCyclotomicElement:
    unit = (Fraction(1),) + (Fraction(0),) * (field.degree - 1)
    return _solve_coordinates(field, a, unit)


def _solve_coordinates(
    field: RationalCyclotomicField,
    a: tuple[Fraction, ...],
    right_hand_side: tuple[Fraction, ...],
) -> RationalCyclotomicElement:
    degree = field.degree
    if not any(a):
        _fail_domain(
            "modular_form.cyclotomic_zero_inverse", "zero has no cyclotomic inverse"
        )
    # Multiplication by a is an invertible Q-linear map exactly when a is
    # nonzero in the field. Solve M_a x = 1 using exact Gauss-Jordan steps.
    columns = []
    for column in range(degree):
        basis = [Fraction(0)] * degree
        basis[column] = Fraction(1)
        raw = [Fraction(0)] * (2 * degree - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(basis):
                raw[i + j] += x * y
        columns.append(_reduce(raw, field))
    matrix = [
        [columns[column][row] for column in range(degree)] + [right_hand_side[row]]
        for row in range(degree)
    ]
    for column in range(degree):
        pivot = next(
            (row for row in range(column, degree) if matrix[row][column]), None
        )
        if pivot is None:
            _fail_domain(
                "modular_form.cyclotomic_zero_inverse", "zero has no cyclotomic inverse"
            )
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [entry / scale for entry in matrix[column]]
        for row in range(degree):
            if row == column:
                continue
            scale = matrix[row][column]
            if scale:
                matrix[row] = [
                    x - scale * y
                    for x, y in zip(matrix[row], matrix[column], strict=True)
                ]
    return _canonical(field, tuple(matrix[index][-1] for index in range(degree)))


def divide(
    left: RationalCyclotomicElement, right: RationalCyclotomicElement
) -> RationalCyclotomicElement:
    degree = MAX_MODULAR_FORM_COEFFICIENT_FIELD_DEGREE
    field, left_values, right_values = _admit(
        left,
        right,
        cost=degree**3 * 4,
        inverse_operand=right,
        include_left_in_inverse_bound=True,
    )
    assert right_values is not None
    return _solve_coordinates(field, right_values, left_values)


__all__ = ["add", "divide", "inverse", "multiply", "subtract"]
