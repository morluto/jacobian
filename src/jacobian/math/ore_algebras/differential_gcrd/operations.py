"""Exact GCRDs for first-order operators with rational constant coefficients."""

from __future__ import annotations

from fractions import Fraction

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras.differential_gcrd._models import (
    DifferentialOperatorGCRDResult,
)
from jacobian.math.ore_algebras.operations import (
    _admit_differential_operator,
    _decode_rf,
    _encode_differential_rf,
)
from jacobian.math.polynomials.values import MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS

_MAX_ORDER = 1
_MAX_INPUT_SCALAR_DIGITS = 20
_MAX_OPERATION_SCALAR_DIGITS = 5 * _MAX_INPUT_SCALAR_DIGITS + 2
_MAX_WORK_UNITS = 32


def _operator(coefficients: dict[int, Fraction]) -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": order,
                    "coefficient": _encode_differential_rf(
                        ({0: coefficient}, {0: Fraction(1)})
                    ),
                }
                for order, coefficient in sorted(coefficients.items())
                if coefficient
            ],
        }
    )


def _coefficient(operator: DifferentialOreOperator, order: int) -> Fraction:
    term = next((term for term in operator.terms if term.order == order), None)
    if term is None:
        return Fraction(0)
    numerator, denominator = _decode_rf(term.coefficient)
    if set(numerator) - {0} or set(denominator) - {0}:
        raise OperationDomainValidationError(
            location=("operator", "terms", order, "coefficient"),
            code="ore_algebra.differential_gcrd_constant_coefficients",
            message="this GCRD slice accepts rational constant coefficients only",
        )
    return numerator.get(0, Fraction(0)) / denominator[0]


def _digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _admit_pair(
    left: DifferentialOreOperator, right: DifferentialOreOperator
) -> tuple[DifferentialOreOperator, DifferentialOreOperator]:
    admitted = (
        _admit_differential_operator(left),
        _admit_differential_operator(right),
    )
    work = 0
    maximum_input_digits = 1
    for label, operator in zip(("left", "right"), admitted, strict=True):
        if operator.order > _MAX_ORDER:
            raise OperationResourceAdmissionError(
                location=(label, "terms"),
                code="ore_algebra.differential_gcrd_order",
                message="this GCRD slice accepts differential order at most one",
            )
        for term in operator.terms:
            numerator, denominator = _decode_rf(term.coefficient)
            if set(numerator) - {0} or set(denominator) - {0}:
                raise OperationDomainValidationError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.differential_gcrd_constant_coefficients",
                    message="this GCRD slice accepts rational constant coefficients only",
                )
            value = numerator.get(0, Fraction(0)) / denominator[0]
            maximum_input_digits = max(maximum_input_digits, _digits(value))
            work += 1

    if maximum_input_digits > _MAX_INPUT_SCALAR_DIGITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_gcrd_scalar_digits",
            message="GCRD input rational constants are limited to 20 decimal digits",
        )
    if 5 * maximum_input_digits + 2 > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_gcrd_output_digits",
            message="GCRD output could exceed the exact rational-function scalar carrier",
        )
    if work > _MAX_WORK_UNITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_gcrd_work",
            message="GCRD inputs exceed the exact coefficient-work envelope",
        )
    if _MAX_OPERATION_SCALAR_DIGITS > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
        raise AssertionError("GCRD output envelope exceeds its scalar carrier")
    return admitted


def differential_operator_gcrd(
    left: DifferentialOreOperator, right: DifferentialOreOperator
) -> DifferentialOperatorGCRDResult:
    """Return the monic GCRD and Bézout factors for order-one QQ[D] inputs.

    Coefficients are rational constants embedded in QQ(x). For two nonzero
    first-order inputs ``a*D+b`` and ``c*D+d``, a nonzero ``b*c-a*d`` gives
    an explicit scalar Bézout identity for one. If it vanishes, both inputs
    are left scalar multiples of the same monic first-order operator. These
    identities also prove maximality in the ambient Ore ring QQ(x)<D>.
    """
    left, right = _admit_pair(left, right)
    left_order, right_order = left.order, right.order

    if left_order < 0 and right_order < 0:
        divisor = _operator({})
        left_cofactor = _operator({})
        right_cofactor = _operator({})
        bezout_left = _operator({})
        bezout_right = _operator({})
    elif left_order < 0 or right_order < 0:
        source = right if left_order < 0 else left
        leading = _coefficient(source, source.order)
        divisor = _operator(
            {
                order: _coefficient(source, order) / leading
                for order in range(source.order + 1)
            }
        )
        left_cofactor = _operator({0: leading}) if left_order >= 0 else _operator({})
        right_cofactor = _operator({0: leading}) if right_order >= 0 else _operator({})
        bezout_left = _operator({0: 1 / leading}) if left_order >= 0 else _operator({})
        bezout_right = (
            _operator({0: 1 / leading}) if right_order >= 0 else _operator({})
        )
    elif left_order == 0 or right_order == 0:
        divisor = _operator({0: Fraction(1)})
        left_cofactor, right_cofactor = left, right
        if left_order == 0:
            bezout_left = _operator({0: 1 / _coefficient(left, 0)})
            bezout_right = _operator({})
        else:
            bezout_left = _operator({})
            bezout_right = _operator({0: 1 / _coefficient(right, 0)})
    else:
        a, b = _coefficient(left, 1), _coefficient(left, 0)
        c, d = _coefficient(right, 1), _coefficient(right, 0)
        determinant = b * c - a * d
        if determinant == 0:
            normalized_constant = b / a
            divisor = _operator({1: Fraction(1), 0: normalized_constant})
            left_cofactor = _operator({0: a})
            right_cofactor = _operator({0: c})
            bezout_left = _operator({0: 1 / a})
            bezout_right = _operator({})
        else:
            divisor = _operator({0: Fraction(1)})
            left_cofactor, right_cofactor = left, right
            bezout_left = _operator({0: c / determinant})
            bezout_right = _operator({0: -a / determinant})

    # The input-dependent 5h+2 admission bound above covers every output scalar.

    return DifferentialOperatorGCRDResult(
        left=left,
        right=right,
        divisor=divisor,
        left_cofactor=left_cofactor,
        right_cofactor=right_cofactor,
        bezout_left=bezout_left,
        bezout_right=bezout_right,
    )


__all__ = ["differential_operator_gcrd"]
