"""Gaussian-rational projective cross ratios."""

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


def _determinant(
    left: GaussianProjectiveLinePoint, right: GaussianProjectiveLinePoint
) -> tuple[Fraction, Fraction]:
    a, b = (coordinate.as_fractions() for coordinate in left.coordinates)
    c, d = (coordinate.as_fractions() for coordinate in right.coordinates)
    return _subtract(_multiply(a, d), _multiply(b, c))


def _integer_digits(value: int) -> int:
    magnitude = abs(value)
    if magnitude < 10:
        return 1
    return (magnitude.bit_length() * 30103) // 100000 + 1


def _cancelled_product_digits(left: Fraction, right: Fraction) -> tuple[int, int]:
    left_num, left_den = abs(left.numerator), left.denominator
    right_num, right_den = abs(right.numerator), right.denominator
    cross_left = gcd(left_num, right_den)
    cross_right = gcd(right_num, left_den)
    return (
        _integer_digits(left_num // cross_left)
        + _integer_digits(right_num // cross_right),
        _integer_digits(left_den // cross_right)
        + _integer_digits(right_den // cross_left),
    )


def _cancelled_sum_digits(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[int, int]:
    left_num, left_den = left
    right_num, right_den = right
    return (
        max(left_num + right_den, right_num + left_den) + 1,
        left_den + right_den,
    )


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

    def _divide_digits(payload: tuple[int, int], modulus: tuple[int, int]) -> int:
        payload_num, payload_den = payload
        modulus_num, modulus_den = modulus
        return max(payload_num + modulus_den, payload_den + modulus_num)

    return max(_divide_digits(real_num, norm), _divide_digits(imag_num, norm))


def _fraction_component_digits(value: Fraction) -> int:
    return max(
        len(format_canonical_integer(abs(value.numerator))),
        len(format_canonical_integer(value.denominator)),
    )


def _gaussian_component_digits(value: tuple[Fraction, Fraction]) -> int:
    return max(_fraction_component_digits(component) for component in value)


def _reject_resource(code: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("first", "second", "third", "fourth"),
        code=f"geometry.gaussian_cross_ratio.{code}",
        message=message,
    )


def _admit_request(request: GaussianCrossRatioSource) -> None:
    """Admit pairwise distinct points and all exact intermediate heights."""

    points = (request.first, request.second, request.third, request.fourth)
    for left_index, left in enumerate(points):
        for right in points[left_index + 1 :]:
            if left.coordinates == right.coordinates:
                raise OperationDomainValidationError(
                    location=("first", "second", "third", "fourth"),
                    code="geometry.gaussian_cross_ratio.points_not_distinct",
                    message="cross-ratio inputs must be pairwise projectively distinct",
                )

    numerator = _multiply(
        _determinant(request.first, request.third),
        _determinant(request.second, request.fourth),
    )
    denominator = _multiply(
        _determinant(request.first, request.fourth),
        _determinant(request.second, request.third),
    )
    # Bound the quotient from the actual determinant products so structurally
    # sparse points are not charged a dense worst-case height.
    product_digits = max(
        _gaussian_component_digits(numerator), _gaussian_component_digits(denominator)
    )
    quotient_digits = 4 * product_digits + 3
    if quotient_digits > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS:
        _reject_resource(
            "intermediate_height_bound",
            "cross-ratio determinant products and quotient exceed the exact intermediate digit bound",
        )
    result_digits = _gaussian_quotient_digit_bound(numerator, denominator)
    if result_digits > MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS:
        if numerator[1] == 0 and denominator[1] == 0 and denominator[0]:
            reduced = numerator[0] / denominator[0]
            if (
                max(
                    _fraction_component_digits(reduced),
                    _fraction_component_digits(Fraction()),
                )
                <= MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS
            ):
                return
        if numerator[0] == 0 and denominator[0] == 0 and denominator[1]:
            reduced = numerator[1] / denominator[1]
            if (
                _fraction_component_digits(reduced)
                <= MAX_GAUSSIAN_RATIONAL_COMPONENT_DIGITS
            ):
                return
        _reject_resource(
            "output_height_bound",
            "cross-ratio output exceeds the Gaussian-rational component bound",
        )


def gaussian_rational_cross_ratio(
    request: GaussianCrossRatioSource,
) -> GaussianRational:
    request_checkpoint("before cross-ratio admission")
    _admit_request(request)
    request_checkpoint("after cross-ratio admission")

    numerator = _multiply(
        _determinant(request.first, request.third),
        _determinant(request.second, request.fourth),
    )
    denominator = _multiply(
        _determinant(request.first, request.fourth),
        _determinant(request.second, request.third),
    )
    if not denominator[0] and not denominator[1]:
        raise OperationDomainValidationError(
            location=("first", "second", "third", "fourth"),
            code="geometry.gaussian_cross_ratio.undefined",
            message="cross-ratio denominator determinants must be nonzero",
        )
    try:
        result = GaussianRational.from_fractions(*_divide(numerator, denominator))
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=("first", "second", "third", "fourth"),
            code="geometry.gaussian_cross_ratio.output_height_bound",
            message="cross-ratio output exceeds the Gaussian-rational component bound",
        ) from exc
    request_checkpoint("after cross-ratio result construction")
    return result


__all__ = ["gaussian_rational_cross_ratio"]
