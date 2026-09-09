"""Gaussian-rational projective cross ratios."""

from fractions import Fraction

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.gaussian_projective_line._models import (
    GaussianCrossRatioRequest,
    GaussianProjectiveLinePoint,
    _divide,
)
from jacobian.math.number_theory.number_fields import GaussianRational


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


def gaussian_rational_cross_ratio(
    request: GaussianCrossRatioRequest,
) -> GaussianRational:
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
    return GaussianRational.from_fractions(*_divide(numerator, denominator))


__all__ = ["gaussian_rational_cross_ratio"]
