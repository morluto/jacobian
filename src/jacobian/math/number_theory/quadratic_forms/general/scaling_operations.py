"""Exact scalar multiplication for rational quadratic forms."""

from __future__ import annotations

from math import gcd

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general.scaling_models import (
    MAX_QUADRATIC_SCALE_AXIS,
    MAX_QUADRATIC_SCALE_SUPPORT,
    QuadraticFormScaleResult,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    QuadraticCrossTerm,
    RationalQuadraticForm,
)

_MAX_COMPONENT = 10**MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1

__all__ = ["scale_rational_quadratic_form"]


def _scaled_factors(
    value: CanonicalRational, factor: CanonicalRational
) -> tuple[int, int, int, int]:
    """Return canceled numerator/denominator factors without multiplying."""

    numerator, denominator = value.as_integer_ratio()
    factor_numerator, factor_denominator = factor.as_integer_ratio()
    if numerator == 0 or factor_numerator == 0:
        return (0, 1, 1, 1)

    cancel_num_den = gcd(abs(numerator), factor_denominator)
    cancel_factor_den = gcd(abs(factor_numerator), denominator)
    return (
        numerator // cancel_num_den,
        factor_numerator // cancel_factor_den,
        denominator // cancel_factor_den,
        factor_denominator // cancel_num_den,
    )


def require_scale_budget(
    form: RationalQuadraticForm, factor: CanonicalRational
) -> None:
    """Admit traversal, coefficient products, and retained exact output.

    Charges the ordered axis length, the retained coefficient support, the
    factor's own width, and the exact numerator and denominator growth of every
    product. Growth is checked with integer division before any product is
    formed, so an over-height coefficient is refused rather than expanded.
    """

    axis = len(form.axis)
    support = len(form.diagonal_coefficients) + len(form.cross_terms)
    if axis > MAX_QUADRATIC_SCALE_AXIS:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.scale_axis_bound",
            message=(
                "quadratic-form scaling axis exceeds the "
                f"{MAX_QUADRATIC_SCALE_AXIS}-coordinate envelope"
            ),
        )
    if support > MAX_QUADRATIC_SCALE_SUPPORT:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.scale_support_bound",
            message=(
                "quadratic-form scaling support exceeds the "
                f"{MAX_QUADRATIC_SCALE_SUPPORT}-term envelope"
            ),
        )
    try:
        require_bounded_rational(
            factor,
            max_digits=MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
            label="quadratic-form scale factor",
        )
    except ValueError as error:
        raise OperationResourceAdmissionError(
            location=("factor",),
            code="quadratic_form.scale_factor_bound",
            message=str(error),
        ) from error

    for coefficient in (
        *form.diagonal_coefficients,
        *(term.coefficient for term in form.cross_terms),
    ):
        left_num, right_num, left_den, right_den = _scaled_factors(coefficient, factor)
        if right_num and abs(left_num) > _MAX_COMPONENT // abs(right_num):
            raise OperationResourceAdmissionError(
                location=("form",),
                code="quadratic_form.scale_coefficient_growth",
                message=(
                    "a scaled numerator would exceed the "
                    f"{MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS}-digit coefficient bound"
                ),
            )
        if left_den > _MAX_COMPONENT // right_den:
            raise OperationResourceAdmissionError(
                location=("form",),
                code="quadratic_form.scale_coefficient_growth",
                message=(
                    "a scaled denominator would exceed the "
                    f"{MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS}-digit coefficient bound"
                ),
            )


def scale_rational_quadratic_form(
    form: RationalQuadraticForm, factor: CanonicalRational
) -> QuadraticFormScaleResult:
    """Return ``factor * Q`` on the same ordered coordinate axis.

    Scaling is coefficientwise over the diagonal and the cross terms, so the
    ordered axis and every coefficient slot are preserved. A cross term whose
    scaled numerator cancels to zero is dropped, which is the only way the
    result can have fewer cross terms than the source.
    """

    require_scale_budget(form, factor)
    diagonal: list[CanonicalRational] = []
    for coefficient in form.diagonal_coefficients:
        left_num, right_num, left_den, right_den = _scaled_factors(coefficient, factor)
        diagonal.append(
            CanonicalRational.from_integer_ratio(
                left_num * right_num, left_den * right_den
            )
        )

    cross_terms: list[QuadraticCrossTerm] = []
    for term in form.cross_terms:
        left_num, right_num, left_den, right_den = _scaled_factors(
            term.coefficient, factor
        )
        numerator = left_num * right_num
        if numerator:
            cross_terms.append(
                QuadraticCrossTerm(
                    left=term.left,
                    right=term.right,
                    coefficient=CanonicalRational.from_integer_ratio(
                        numerator, left_den * right_den
                    ),
                )
            )

    scaled = RationalQuadraticForm(
        axis=form.axis,
        diagonal_coefficients=tuple(diagonal),
        cross_terms=tuple(cross_terms),
    )
    return QuadraticFormScaleResult._from_kernel(
        source_form=form, factor=factor, form=scaled
    )
