"""Gaussian-rational projective cross ratios."""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian._execution import request_checkpoint
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.gaussian_projective_line._models import (
    GaussianCrossRatioSource,
    GaussianProjectiveLinePoint,
    _divide,
)
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.number_theory.number_fields.values import (
    MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS,
)

# The exact kernel is a fixed number of Gaussian additions and multiplications.
# This bound is deliberately on coefficient height, not transport bytes.  It
# keeps every Fraction intermediate below the canonical integer envelope before
# any arithmetic starts.  The Gaussian-rational result has its own 4,096-digit
# component contract and is checked when it is constructed.
MAX_CROSS_RATIO_INTERMEDIATE_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


def _subtract(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] - right[0], left[1] - right[1]


def _multiply(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[
        1
    ] * right[0]


def _fraction_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _gaussian_component_digits(value: tuple[Fraction, Fraction]) -> int:
    return max(_fraction_component_digits(component) for component in value)


def _exceeds_intermediate_digits(digits: int) -> bool:
    return 4 * digits + 3 > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS


def _reject_resource(code: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("first", "second", "third", "fourth"),
        code=f"geometry.gaussian_cross_ratio.{code}",
        message=message,
    )


def _gaussian_multiply_digit_bound(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> int:
    return (
        2
        * (
            _gaussian_component_digits(left)
            + _gaussian_component_digits(right)
        )
        + 1
    )


def _admitted_multiply(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    if _gaussian_multiply_digit_bound(left, right) > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS:
        _reject_resource(
            "intermediate_height_bound",
            "cross-ratio determinant products and quotient exceed the exact intermediate digit bound",
        )
    product = _multiply(left, right)
    if _gaussian_component_digits(product) > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS:
        _reject_resource(
            "intermediate_height_bound",
            "cross-ratio determinant products and quotient exceed the exact intermediate digit bound",
        )
    return product


def _admitted_determinant(
    left: GaussianProjectiveLinePoint, right: GaussianProjectiveLinePoint
) -> tuple[Fraction, Fraction]:
    a, b = (coordinate.as_fractions() for coordinate in left.coordinates)
    c, d = (coordinate.as_fractions() for coordinate in right.coordinates)
    ad = _admitted_multiply(a, d)
    bc = _admitted_multiply(b, c)
    difference = _subtract(ad, bc)
    if _gaussian_component_digits(difference) > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS:
        _reject_resource(
            "intermediate_height_bound",
            "cross-ratio determinant products and quotient exceed the exact intermediate digit bound",
        )
    return difference


@dataclass(frozen=True, slots=True)
class _CrossRatioPlan:
    """Request-scoped products and the admitted quotient, computed once."""

    numerator: tuple[Fraction, Fraction]
    denominator: tuple[Fraction, Fraction]
    value: GaussianRational


def _admit_request(request: GaussianCrossRatioSource) -> _CrossRatioPlan:
    """Admit distinctness, product height, and the exact quotient once."""

    points = (request.first, request.second, request.third, request.fourth)
    for left_index, left in enumerate(points):
        for right in points[left_index + 1 :]:
            if left.coordinates == right.coordinates:
                raise OperationDomainValidationError(
                    location=("first", "second", "third", "fourth"),
                    code="geometry.gaussian_cross_ratio.points_not_distinct",
                    message="cross-ratio inputs must be pairwise projectively distinct",
                )

    numerator = _admitted_multiply(
        _admitted_determinant(request.first, request.third),
        _admitted_determinant(request.second, request.fourth),
    )
    denominator = _admitted_multiply(
        _admitted_determinant(request.first, request.fourth),
        _admitted_determinant(request.second, request.third),
    )
    product_digits = max(
        _gaussian_component_digits(numerator), _gaussian_component_digits(denominator)
    )
    if _exceeds_intermediate_digits(product_digits):
        _reject_resource(
            "intermediate_height_bound",
            "cross-ratio determinant products and quotient exceed the exact intermediate digit bound",
        )
    if not denominator[0] and not denominator[1]:
        raise OperationDomainValidationError(
            location=("first", "second", "third", "fourth"),
            code="geometry.gaussian_cross_ratio.undefined",
            message="cross-ratio denominator determinants must be nonzero",
        )
    quotient = _divide(numerator, denominator)
    if _gaussian_component_digits(quotient) > MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS:
        _reject_resource(
            "output_height_bound",
            "cross-ratio output exceeds the Gaussian-rational component bound",
        )
    return _CrossRatioPlan(
        numerator=numerator,
        denominator=denominator,
        value=GaussianRational.from_fractions(*quotient),
    )


def gaussian_rational_cross_ratio(
    request: GaussianCrossRatioSource,
) -> GaussianRational:
    request_checkpoint("before cross-ratio admission")
    plan = _admit_request(request)
    request_checkpoint("after cross-ratio admission")
    request_checkpoint("after cross-ratio result construction")
    return plan.value


__all__ = ["gaussian_rational_cross_ratio"]
