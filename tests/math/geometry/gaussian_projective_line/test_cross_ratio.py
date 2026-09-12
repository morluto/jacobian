"""Gaussian-rational projective-line tests."""

from fractions import Fraction
from itertools import permutations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
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
    request = GaussianCrossRatioSource(
        first=_point(_z(0), _z(1)),
        second=_point(_z(0), _z(1)),
        third=_point(_z(1), _z(1)),
        fourth=_point(_z(1), _z(0)),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        gaussian_rational_cross_ratio(request)
    assert (
        error.value.errors()[0]["type"]
        == "geometry.gaussian_cross_ratio.points_not_distinct"
    )


def test_coincident_points_reach_operation_admission_on_dispatch() -> None:
    payload = {
        "first": {
            "coordinates": [
                {
                    "real": {"num": "0", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
        "second": {
            "coordinates": [
                {
                    "real": {"num": "0", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
        "third": {
            "coordinates": [
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
        "fourth": {
            "coordinates": [
                {
                    "real": {"num": "1", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
                {
                    "real": {"num": "0", "den": "1"},
                    "imaginary": {"num": "0", "den": "1"},
                },
            ]
        },
    }
    with pytest.raises(OperationDomainValidationError) as error:
        invoke_operation(
            "geometry.projective_line.cross_ratio.gaussian_rational.compute",
            payload,
            Catalog.open(),
        )
    assert not isinstance(error.value, OperationRequestValidationError)
    assert (
        error.value.errors()[0]["type"]
        == "geometry.gaussian_cross_ratio.points_not_distinct"
    )


def test_sparse_large_coordinate_cross_ratio_is_admitted() -> None:
    large = 10**255 + 1
    request = GaussianCrossRatioSource(
        first=_point(
            GaussianRational.from_fractions(Fraction(large), Fraction()),
            _z(1),
        ),
        second=_point(_z(0), _z(1)),
        third=_point(_z(1), _z(1)),
        fourth=_point(_z(1), _z(0)),
    )
    result = gaussian_rational_cross_ratio(request)
    assert result.as_fractions() == (Fraction(-(10**255)), Fraction())


def test_output_height_is_admitted_before_result_construction() -> None:
    large = 10**2100
    request = GaussianCrossRatioSource(
        first=_point(_z(0), _z(1)),
        second=_point(_z(1), _z(1)),
        third=_point(
            GaussianRational.from_fractions(Fraction(large), Fraction()),
            _z(1),
        ),
        fourth=_point(
            GaussianRational.from_fractions(Fraction(large + 1), Fraction()),
            _z(1),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        gaussian_rational_cross_ratio(request)
    assert (
        error.value.errors()[0]["type"]
        == "geometry.gaussian_cross_ratio.output_height_bound"
    )


def test_mixed_denominator_output_height_is_rejected_before_quotient() -> None:
    q = 10**1499 + 7
    r = q + 1
    request = GaussianCrossRatioSource(
        first=_point(_z(1), _z(0)),
        second=_point(_z(0), _z(1)),
        third=_point(_z(1), _z(1)),
        fourth=_point(
            _z(1),
            GaussianRational.from_fractions(Fraction(1, q), Fraction(1, r)),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        gaussian_rational_cross_ratio(request)
    assert (
        error.value.errors()[0]["type"]
        == "geometry.gaussian_cross_ratio.output_height_bound"
    )


def test_large_harmonic_quadruple_cancels_to_minus_one() -> None:
    scale = 10**4095
    request = GaussianCrossRatioSource(
        first=_point(_z(0), _z(1)),
        second=_point(_z(1), _z(0)),
        third=_point(
            GaussianRational.from_fractions(Fraction(scale), Fraction()),
            _z(1),
        ),
        fourth=_point(
            GaussianRational.from_fractions(Fraction(-scale), Fraction()),
            _z(1),
        ),
    )
    result = gaussian_rational_cross_ratio(request)
    assert result.as_fractions() == (Fraction(-1), Fraction())


def test_independent_coordinate_products_are_refused_before_multiplication() -> None:
    scale = 10**4095
    request = GaussianCrossRatioSource(
        first=_point(_z(1), GaussianRational.from_fractions(Fraction(scale + 3), Fraction(scale + 5))),
        second=_point(
            _z(1),
            GaussianRational.from_fractions(Fraction(scale + 11), Fraction(scale + 13)),
        ),
        third=_point(
            _z(1),
            GaussianRational.from_fractions(
                Fraction(2 * scale + 17), Fraction(3 * scale + 19)
            ),
        ),
        fourth=_point(
            _z(1),
            GaussianRational.from_fractions(
                Fraction(5 * scale + 23), Fraction(7 * scale + 29)
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        gaussian_rational_cross_ratio(request)
    assert (
        error.value.errors()[0]["type"]
        == "geometry.gaussian_cross_ratio.intermediate_height_bound"
    )
