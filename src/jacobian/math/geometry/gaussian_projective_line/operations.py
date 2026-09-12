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


def _integer_digits(value: int) -> int:
    magnitude = abs(value)
    if magnitude < 10:
        return 1
    return (magnitude.bit_length() * 30103) // 100000 + 1


def _cancelled_product_digits(left: Fraction, right: Fraction) -> tuple[int, int, int]:
    left_num, left_den = abs(left.numerator), left.denominator
    right_num, right_den = abs(right.numerator), right.denominator
    cross_left = gcd(left_num, right_den)
    cross_right = gcd(right_num, left_den)
    cancelled_den = (left_den // cross_right) * (right_den // cross_left)
    return (
        _integer_digits(left_num // cross_left)
        + _integer_digits(right_num // cross_right),
        _integer_digits(cancelled_den),
        cancelled_den,
    )


def _cancelled_sum_digits(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> tuple[int, int, int]:
    left_num, left_den_digits, left_den = left
    right_num, right_den_digits, right_den = right
    common = gcd(left_den, right_den)
    common_digits = _integer_digits(common)
    left_scale_digits = right_den_digits - common_digits
    right_scale_digits = left_den_digits - common_digits
    if left_scale_digits < 0:
        left_scale_digits = 0
    if right_scale_digits < 0:
        right_scale_digits = 0
    lcm_digits = left_den_digits + right_scale_digits
    if lcm_digits < max(left_den_digits, right_den_digits):
        lcm_digits = max(left_den_digits, right_den_digits)
    return (
        max(left_num + left_scale_digits, right_num + right_scale_digits) + 1,
        lcm_digits,
        (left_den // common) * right_den,
    )


def _fraction_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _gaussian_component_digits(value: tuple[Fraction, Fraction]) -> int:
    return max(_fraction_component_digits(component) for component in value)


def _gaussian_quotient_digit_bound(
    numerator: tuple[Fraction, Fraction],
    denominator: tuple[Fraction, Fraction],
) -> int:
    """Bound real/imaginary component digits of a Gaussian quotient."""

    real, imag = numerator
    denom_real, denom_imag = denominator
    real_num = _cancelled_sum_digits(
        _cancelled_product_digits(real, denom_real),
        _cancelled_product_digits(imag, denom_imag),
    )
    imag_num = _cancelled_sum_digits(
        _cancelled_product_digits(imag, denom_real),
        _cancelled_product_digits(real, denom_imag),
    )
    norm = _cancelled_sum_digits(
        _cancelled_product_digits(denom_real, denom_real),
        _cancelled_product_digits(denom_imag, denom_imag),
    )

    def _divide_digits(
        payload: tuple[int, int, int], modulus: tuple[int, int, int]
    ) -> int:
        payload_num, payload_den, _payload_value = payload
        modulus_num, modulus_den, _modulus_value = modulus
        return max(payload_num + modulus_den, payload_den + modulus_num)

    coarse = max(_divide_digits(real_num, norm), _divide_digits(imag_num, norm))
    if coarse <= MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS:
        return coarse
    return _gaussian_component_digits(_divide(numerator, denominator))


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
    real, imag = left
    other_real, other_imag = right
    real_part = _cancelled_sum_digits(
        _cancelled_product_digits(real, other_real),
        _cancelled_product_digits(imag, other_imag),
    )
    imag_part = _cancelled_sum_digits(
        _cancelled_product_digits(real, other_imag),
        _cancelled_product_digits(imag, other_real),
    )
    return max(*real_part[:2], *imag_part[:2])


def _admitted_multiply(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    bound = _gaussian_multiply_digit_bound(left, right)
    if bound > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS or _exceeds_intermediate_digits(
        bound
    ):
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
    if (
        _gaussian_quotient_digit_bound(numerator, denominator)
        > MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS
    ):
        _reject_resource(
            "output_height_bound",
            "cross-ratio output exceeds the Gaussian-rational component bound",
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
