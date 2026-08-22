"""Tests for pinned distance to pair-spanned line profiles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._pinned_distances import (
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(a: int, b: int) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational.from_integer_ratio(a[0], a[1]),
        y=CanonicalRational.from_integer_ratio(b[0], b[1]),
    )


class TestKnownAnswers:
    def test_unit_square_profile(self):
        anchor = _point((0, 1), (0, 1))
        points = (
            _point((0, 1), (0, 1)),
            _point((1, 1), (0, 1)),
            _point((0, 1), (1, 1)),
            _point((1, 1), (1, 1)),
        )
        result = compute_pinned_distances(
            PinnedDistanceRequest(anchor=anchor, points=points)
        )
        assert result.distinct_line_count == 6
        squared = {
            (
                entry.squared_distance_numerator,
                entry.squared_distance_denominator,
            )
            for entry in result.lines
        }
        assert ("0", "1") in squared
        assert ("1", "1") in squared
        assert ("1", "2") in squared
        assert result.min_squared_distance is not None
        assert result.min_squared_distance.squared_distance_numerator == "0"
        pairs = {pair for entry in result.lines for pair in entry.source_pairs}
        assert pairs == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}

    def test_duplicate_points_rejected(self):
        anchor = _point((0, 1), (0, 1))
        with pytest.raises(ValidationError, match="unique"):
            PinnedDistanceRequest(
                anchor=anchor,
                points=(
                    _point((1, 1), (0, 1)),
                    _point((1, 1), (0, 1)),
                ),
            )


class TestAboveConversionLimit:
    def test_squared_distance_parsing_above_4300_digits(self):
        """Squared-distance numerators beyond CPython's 4300-digit conversion
        limit must parse through the canonical integer helper."""
        big_den = 10**2047

        def reciprocal_point(a: int, b: int) -> RationalPoint2D:
            return RationalPoint2D(
                x=CanonicalRational.from_integer_ratio(1, big_den + a),
                y=CanonicalRational.from_integer_ratio(1, big_den + b),
            )

        request = PinnedDistanceRequest(
            anchor=reciprocal_point(3, 7),
            points=(reciprocal_point(11, 13), reciprocal_point(17, 19)),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count >= 1
        widest = max(
            len(entry.squared_distance_numerator) for entry in result.lines
        )
        assert widest > 4300
        # Result revalidation replays the minimum through canonical parsing.
        PinnedDistanceResult.model_validate(result.model_dump())
