"""Tests for pinned-distance profiles over pair-spanned lines."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._pinned_distances import (
    LineDistanceEntry,
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(x_num: str, x_den: str, y_num: str, y_den: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=x_num, den=x_den),
        y=CanonicalRational(num=y_num, den=y_den),
    )


class TestPinnedDistanceProfile:
    def test_known_answer_unit_square(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point("0", "1", "0", "1"),
            points=(
                _point("0", "1", "0", "1"),
                _point("1", "1", "0", "1"),
                _point("0", "1", "1", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        # Lines x=0, y=0 pass through the anchor at distance 0; the diagonal
        # x+y=1 sits at squared distance 1/2.
        distances = sorted(
            entry.squared_distance.as_fraction() for entry in result.lines
        )
        assert distances == [Fraction(0), Fraction(0), Fraction(1, 2)]

    def test_distance_entries_are_canonical_rationals(self) -> None:
        """Squared distances carry the domain-owned canonical rational."""
        anchor = _point("0", "1", "0", "1")
        points = (
            _point("3", "1", "0", "1"),
            _point("0", "1", "4", "1"),
            _point("5", "1", "5", "1"),
        )
        result = compute_pinned_distances(
            PinnedDistanceRequest(anchor=anchor, points=points)
        )
        for entry in result.lines:
            assert isinstance(entry.squared_distance, CanonicalRational)

    def test_unrepresentable_squared_distance_rejected_at_admission(self) -> None:
        """Distinct 32k-digit denominators would produce a squared distance
        no canonical rational accepts; the coordinate-height bound rejects
        the configuration at the typed boundary instead."""
        big = format_canonical_integer(10**32000 + 1)
        with pytest.raises(ValidationError, match="point-set coordinate"):
            PinnedDistanceRequest(
                anchor=_point("0", "1", "0", "1"),
                points=(
                    _point("1", big, "0", "1"),
                    _point("0", "1", "1", big),
                    _point("2", "1", "2", "1"),
                ),
            )

    def test_minimum_must_be_a_complete_ledger_entry(self) -> None:
        """A separate minimum entry with forged source pairs is rejected even
        when its scalar distance equals the true minimum."""
        request = PinnedDistanceRequest(
            anchor=_point("0", "1", "0", "1"),
            points=(
                _point("1", "1", "0", "1"),
                _point("0", "1", "1", "1"),
                _point("1", "1", "1", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        payload = result.model_dump(mode="json")
        # Forge the minimum: keep its scalar distance but replace source pairs.
        forged_min = {
            "squared_distance": {"num": "1", "den": "1"},
            "source_pairs": [[98, 99]],
        }
        payload["min_squared_distance"] = forged_min
        with pytest.raises(ValidationError, match="complete line-ledger"):
            PinnedDistanceResult.model_validate(payload)
