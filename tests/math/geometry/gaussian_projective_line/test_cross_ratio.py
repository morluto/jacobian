"""Gaussian-rational projective-line tests."""

from fractions import Fraction
from itertools import permutations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
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


def _source(
    points: tuple[GaussianProjectiveLinePoint, ...],
) -> GaussianCrossRatioSource:
    return GaussianCrossRatioSource(
        first=points[0], second=points[1], third=points[2], fourth=points[3]
    )


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


def test_harmonic_quadruple_returns_minus_one() -> None:
    result = gaussian_rational_cross_ratio(
        GaussianCrossRatioSource(
            first=_point(_z(0), _z(1)),
            second=_point(_z(1), _z(0)),
            third=_point(_z(1), _z(1)),
            fourth=_point(_z(-1), _z(1)),
        )
    )
    assert result.as_fractions() == (Fraction(-1), Fraction())


def test_projective_matrix_action_preserves_cross_ratio() -> None:
    source_points = (
        _point(_z(0), _z(1)),
        _point(_z(1), _z(1)),
        _point(_z(2), _z(1)),
        _point(_z(1), _z(0)),
    )

    def transform(point: GaussianProjectiveLinePoint) -> GaussianProjectiveLinePoint:
        x, y = (coordinate.as_fractions() for coordinate in point.coordinates)
        # [[2, 1], [1, 1]] has determinant 1 and is therefore invertible.
        return _point(
            GaussianRational.from_fractions(2 * x[0] + y[0], 2 * x[1] + y[1]),
            GaussianRational.from_fractions(x[0] + y[0], x[1] + y[1]),
        )

    result = gaussian_rational_cross_ratio(_source(source_points))
    transformed = gaussian_rational_cross_ratio(
        _source(tuple(transform(point) for point in source_points))
    )
    assert transformed == result


def test_permutations_have_the_six_cross_ratio_values() -> None:
    points = (
        _point(_z(0), _z(1)),
        _point(_z(1), _z(1)),
        _point(_z(2), _z(1)),
        _point(_z(1), _z(0)),
    )
    values = {
        gaussian_rational_cross_ratio(_source(ordering)).as_fractions()
        for ordering in permutations(points)
    }
    assert values == {
        (Fraction(2), Fraction()),
        (Fraction(1, 2), Fraction()),
        (Fraction(-1), Fraction()),
    }


def test_coincident_points_are_rejected_before_division() -> None:
    with pytest.raises(ValidationError, match="pairwise projectively distinct"):
        GaussianCrossRatioSource(
            first=_point(_z(0), _z(1)),
            second=_point(_z(0), _z(1)),
            third=_point(_z(1), _z(1)),
            fourth=_point(_z(1), _z(0)),
        )


def test_cross_ratio_intermediate_height_is_admitted_semantically() -> None:
    large = 10**255 + 1
    request = GaussianCrossRatioSource(
        first=_point(
            GaussianRational.from_fractions(Fraction(large), Fraction(1)),
            _z(1),
        ),
        second=_point(_z(0), _z(1)),
        third=_point(_z(1), _z(1)),
        fourth=_point(_z(1), _z(0)),
    )
    with pytest.raises(OperationResourceAdmissionError, match="intermediate digit"):
        gaussian_rational_cross_ratio(request)
