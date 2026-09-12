"""Gaussian-rational projective cross ratios."""

from fractions import Fraction
from typing import NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian._execution import request_checkpoint
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


def _fraction_component_digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _point_component_digits(point: GaussianProjectiveLinePoint) -> int:
    return max(
        _fraction_component_digits(component)
        for coordinate in point.coordinates
        for component in coordinate.as_fractions()
    )


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

    source_digits = max(_point_component_digits(point) for point in points)
    # A complex determinant is one subtraction of two complex products; the
    # following products and quotient are then bounded independently.  The +1
    # terms cover a carry from adding two signed integers.
    determinant_digits = 8 * source_digits + 3
    product_digits = 4 * determinant_digits + 1
    # Complex division forms a norm (one more sum of products), then multiplies
    # the numerator by that norm's denominator.  Both the norm and the complex
    # numerator can reach 4*product_digits+1, so the final Fraction components
    # need twice that height.
    quotient_digits = 8 * product_digits + 3
    if quotient_digits > MAX_CROSS_RATIO_INTERMEDIATE_DIGITS:
        _reject_resource(
            "intermediate_height_bound",
            "cross-ratio determinant products and quotient exceed the exact intermediate digit bound",
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
