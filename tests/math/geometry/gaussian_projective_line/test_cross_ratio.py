"""Gaussian-rational projective-line tests."""

from fractions import Fraction

from jacobian.math.geometry.gaussian_projective_line._models import (
    GaussianCrossRatioSource,
    GaussianProjectiveLinePoint,
)
from jacobian.math.geometry.gaussian_projective_line.operations import (
    gaussian_rational_cross_ratio,
)
from jacobian.math.number_theory.number_fields import GaussianRational


def _z(real: int, imaginary: int = 0) -> GaussianRational:
    return GaussianRational.from_fractions(Fraction(real), Fraction(imaginary))


def _point(
    left: GaussianRational, right: GaussianRational
) -> GaussianProjectiveLinePoint:
    return GaussianProjectiveLinePoint(coordinates=(left, right))


def test_points_use_first_nonzero_coordinate_normalization() -> None:
    point = _point(_z(2, 2), _z(4, 0))
    assert point.coordinates[0] == GaussianRational.one()
    assert point.coordinates[1].as_fractions() == (Fraction(1), Fraction(-1))


def test_rational_cross_ratio_is_exact_gaussian_scalar() -> None:
    result = gaussian_rational_cross_ratio(
        GaussianCrossRatioSource(
            first=_point(_z(0), _z(1)),
            second=_point(_z(1), _z(1)),
            third=_point(_z(2), _z(1)),
            fourth=_point(_z(1), _z(0)),
        )
    )
    assert result.as_fractions() == (Fraction(2), Fraction())


def test_cross_ratio_can_be_nonreal() -> None:
    result = gaussian_rational_cross_ratio(
        GaussianCrossRatioSource(
            first=_point(_z(0), _z(1)),
            second=_point(_z(1), _z(1)),
            third=_point(_z(0, 1), _z(1)),
            fourth=_point(_z(1), _z(0)),
        )
    )
    assert result.imaginary.num != 0
