"""Tests for circumradius profiles and forbidden-pattern screening."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    LabelledPoint2D,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import circumradius_profile


def _reciprocal_labelled(label: str, first: int, second: int, digits: int) -> LabelledPoint2D:
    scale = 10 ** (digits - 1)
    return LabelledPoint2D(
        label=label,
        point=RationalPoint2D(
            x=CanonicalRational.from_integer_ratio(1, scale + first),
            y=CanonicalRational.from_integer_ratio(1, scale + second),
        ),
    )


def _ratio_point(label: str, index: int, digits: int) -> LabelledPoint2D:
    """Deterministic distinct coordinates whose components have exactly `digits` digits."""
    shift = index * digits
    return LabelledPoint2D(
        label=label,
        point=RationalPoint2D(
            x=CanonicalRational.from_integer_ratio(
                10 ** (digits - 1) + shift + index + 1,
                10**digits - shift - 2 * index - 7,
            ),
            y=CanonicalRational.from_integer_ratio(
                10**digits - shift - 2 * index - 9,
                10 ** (digits - 1) + shift + index + 3,
            ),
        ),
    )


class TestCircumradiusProfile:
    def test_unit_square_known_answer(self):
        def point(x: int, y: int) -> LabelledPoint2D:
            return LabelledPoint2D(
                label=f"P{x}{y}",
                point=RationalPoint2D(
                    x=CanonicalRational.from_integer_ratio(x, 1),
                    y=CanonicalRational.from_integer_ratio(y, 1),
                ),
            )

        result = circumradius_profile(
            CircumradiusProfileRequest(
                points=(point(0, 0), point(1, 0), point(0, 1))
            )
        )
        assert result.triple_count == 1
        entry = result.entries[0]
        assert not entry.collinear
        assert entry.squared_circumradius is not None
        assert (
            entry.squared_circumradius.num == "1"
            and entry.squared_circumradius.den == "2"
        )

    def test_collinear_triple_flagged(self):
        def point(x: int, y: int) -> LabelledPoint2D:
            return LabelledPoint2D(
                label=f"P{x}_{y}",
                point=RationalPoint2D(
                    x=CanonicalRational.from_integer_ratio(x, 1),
                    y=CanonicalRational.from_integer_ratio(y, 1),
                ),
            )

        result = circumradius_profile(
            CircumradiusProfileRequest(
                points=(point(0, 0), point(1, 0), point(2, 0))
            )
        )
        assert result.entries[0].collinear
        assert result.entries[0].squared_circumradius is None

    def test_result_overflow_inputs_rejected_at_admission(self):
        """Coordinates that pass a naive per-component cap but whose squared
        circumradius exceeds the canonical limit must be rejected by the request
        model instead of failing during result construction."""
        points = tuple(
            _reciprocal_labelled(label, first, second, 4096)
            for label, first, second in (("a", 3, 7), ("b", 11, 13), ("c", 17, 19))
        )
        with pytest.raises(ValidationError, match="819-digit bound"):
            CircumradiusProfileRequest(points=points)

    def test_boundary_digit_coordinates_succeed(self):
        """The largest admitted coordinate size must still return exact values."""
        points = (
            _ratio_point("a", 0, 819),
            _ratio_point("b", 1, 819),
            _ratio_point("c", 2, 819),
        )
        result = circumradius_profile(CircumradiusProfileRequest(points=points))
        assert result.triple_count == 1
        entry = result.entries[0]
        assert not entry.collinear
        assert entry.squared_circumradius is not None
        assert len(entry.squared_circumradius.num) <= 32_768
        assert len(entry.squared_circumradius.den) <= 32_768


class TestCircumradiusAggregateBudget:
    def test_large_profile_with_boundary_digits_rejected(self):
        """The 24-point reciprocal construction at the 819-digit boundary
        would emit a ~46 MB exact profile; the aggregate output budget
        couples point count and coordinate size to reject it."""
        points = tuple(_ratio_point(f"P{i}", i, 819) for i in range(24))
        with pytest.raises(ValidationError, match="aggregate output"):
            CircumradiusProfileRequest(points=points)

    def test_small_coordinate_full_profile_still_admitted(self):
        """Small-coordinate configurations keep the full 24-point profile."""
        points = tuple(
            LabelledPoint2D(
                label=f"P{i}",
                point=RationalPoint2D(
                    x=CanonicalRational.from_integer_ratio(i, 1),
                    y=CanonicalRational.from_integer_ratio(i * i + 1, 1),
                ),
            )
            for i in range(24)
        )
        request = CircumradiusProfileRequest(points=points)
        result = circumradius_profile(request)
        assert result.triple_count == 2024
