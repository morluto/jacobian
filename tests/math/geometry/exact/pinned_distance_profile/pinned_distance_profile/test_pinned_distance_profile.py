from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.pinned_distance_profile._models import (
    PinnedDistanceProfileRequest,
)
from jacobian.math.geometry.exact.pinned_distance_profile.operations import (
    compute_pinned_distance_profile,
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


def test_three_collinear() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0]), ("c", [2, 0])])
    result = compute_pinned_distance_profile(config)
    assert len(result.profiles) == 3

    # Check point a: b at distance 1, c at distance 4
    a_profile = result.profiles[0]
    assert a_profile.entries[0].squared_distance.as_fraction() == Fraction(1)
    assert a_profile.entries[0].target_labels == ("b",)
    assert a_profile.entries[1].squared_distance.as_fraction() == Fraction(4)
    assert a_profile.entries[1].target_labels == ("c",)


def test_two_points() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 1])])
    result = compute_pinned_distance_profile(config)
    assert len(result.profiles) == 2
    # d(a,b) = 1^2 + 1^2 = 2
    assert result.profiles[0].entries[0].squared_distance.as_fraction() == Fraction(2)


def test_same_distance() -> None:
    config = _config([("center", [0, 0]), ("a", [1, 0]), ("b", [-1, 0])])
    result = compute_pinned_distance_profile(config)
    # From center: a and b both at distance 1
    center_profile = result.profiles[0]
    assert len(center_profile.entries) == 1  # One distance class
    assert center_profile.entries[0].squared_distance.as_fraction() == Fraction(1)
    assert set(center_profile.entries[0].target_labels) == {"a", "b"}


def test_result_preserves_source() -> None:
    config = _config([("a", [0, 0]), ("b", [1, 0])])
    result = compute_pinned_distance_profile(config)
    assert result.configuration == config


def test_derived_squared_distance_digit_growth_is_rejected() -> None:
    config = _config([("a", [0]), ("b", [10**256])])

    with pytest.raises(OperationDomainValidationError, match="256"):
        compute_pinned_distance_profile(config)


def test_request_rejects_coordinate_height_before_execution() -> None:
    config = _config([("a", [0]), ("b", [10**256])])

    with pytest.raises(ValidationError, match="256"):
        PinnedDistanceProfileRequest(configuration=config)
