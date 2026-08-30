from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.triangle_area_profile.operations import (
    compute_triangle_area_profile,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def _config(points: Sequence[tuple[str, Sequence[int]]]) -> PointConfiguration:
    return PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=_label,
                coordinates=tuple(_cr(x) for x in coords),
            )
            for _label, coords in points
        )
    )


def test_unit_square() -> None:
    config = _config(
        [
            ("a", [0, 0]),
            ("b", [1, 0]),
            ("c", [1, 1]),
            ("d", [0, 1]),
        ]
    )
    result = compute_triangle_area_profile(config)
    assert len(result.entries) == 4  # C(4,3) = 4
    # All triangles have area 1/2
    for entry in result.entries:
        assert entry.area.as_fraction() == Fraction(1, 2)


def test_collinear() -> None:
    config = _config(
        [
            ("a", [0, 0]),
            ("b", [1, 0]),
            ("c", [2, 0]),
        ]
    )
    result = compute_triangle_area_profile(config)
    assert len(result.entries) == 1
    assert result.entries[0].area.as_fraction() == Fraction(0)


def test_right_triangle() -> None:
    config = _config(
        [
            ("a", [0, 0]),
            ("b", [3, 0]),
            ("c", [0, 4]),
        ]
    )
    result = compute_triangle_area_profile(config)
    assert len(result.entries) == 1
    assert result.entries[0].area.as_fraction() == Fraction(6)


def test_result_preserves_source() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0]), ("c", [0, 1])])
    result = compute_triangle_area_profile(config)
    assert result.configuration == config


def test_derived_area_must_fit_the_canonical_rational_carrier() -> None:
    denominator = 10**20_000
    config = PointConfiguration(
        points=(
            LabelledRationalPoint(label="a", coordinates=(_cr(0), _cr(0))),
            LabelledRationalPoint(label="b", coordinates=(_cr(1, denominator), _cr(0))),
            LabelledRationalPoint(label="c", coordinates=(_cr(0), _cr(1, denominator))),
        )
    )

    with pytest.raises(OperationDomainValidationError, match="derived triangle area"):
        compute_triangle_area_profile(config)


@pytest.mark.parametrize("dimension", [1, 3])
def test_nonplanar_configuration_is_rejected(dimension: int) -> None:
    config = _config(
        [
            ("a", [0] * dimension),
            ("b", [1] * dimension),
            ("c", [2] * dimension),
        ]
    )

    with pytest.raises(OperationDomainValidationError, match="exactly two"):
        compute_triangle_area_profile(config)
