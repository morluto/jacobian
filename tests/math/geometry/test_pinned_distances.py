"""Tests for exact pinned distances to pair-spanned lines."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._pinned_distances import (
    LineDistanceEntry,
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(x: str, y: str) -> dict:
    return {
        "x": {"num": x, "den": "1"},
        "y": {"num": y, "den": "1"},
    }


def _request(anchor: dict, points: tuple[dict, ...]) -> PinnedDistanceRequest:
    return PinnedDistanceRequest.model_validate(
        {"anchor": anchor, "points": list(points)}
    )


class TestPinnedDistances:
    def test_unit_square_lines_are_distinct(self):
        request = _request(
            _point("0", "0"),
            (
                _point("0", "0"),
                _point("1", "0"),
                _point("1", "1"),
                _point("0", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 6
        distances = {entry.squared_distance for entry in result.lines}
        assert CanonicalRational(num="0", den="1") in distances
        assert CanonicalRational(num="1", den="2") in distances

    def test_same_distance_lines_are_not_merged(self):
        # The axes x=0 and y=0 both lie at squared distance zero from the
        # anchor, but they are distinct lines and must stay distinct entries.
        request = _request(
            _point("0", "0"),
            (
                _point("0", "0"),
                _point("1", "0"),
                _point("0", "1"),
            ),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 3
        zeros = [
            entry
            for entry in result.lines
            if entry.squared_distance == CanonicalRational(num="0", den="1")
        ]
        assert len(zeros) == 2

    def test_entries_are_canonical_rationals(self):
        request = _request(
            _point("0", "0"),
            (_point("-1", "0"), _point("1", "1")),
        )
        result = compute_pinned_distances(request)
        assert all(
            isinstance(entry.squared_distance, CanonicalRational)
            for entry in result.lines
        )
        payload = result.model_dump()
        assert set(payload["lines"][0]["squared_distance"]) == {"num", "den"}

    def test_min_squared_distance(self):
        request = _request(
            _point("0", "0"),
            (_point("2", "0"), _point("0", "3"), _point("1", "1")),
        )
        result = compute_pinned_distances(request)
        minimum = min(entry.squared_distance.as_fraction() for entry in result.lines)
        assert result.min_squared_distance is not None
        assert result.min_squared_distance.squared_distance.as_fraction() == minimum

    def test_source_pairs_replay_against_retained_points(self):
        request = _request(
            _point("0", "0"),
            (_point("1", "0"), _point("0", "1"), _point("1", "1")),
        )
        result = compute_pinned_distances(request)
        PinnedDistanceResult.model_validate(result.model_dump())
        anchor = request.anchor
        points = request.points
        for entry in result.lines:
            for i, j in entry.source_pairs:
                xi, yi = points[i].x.as_fraction(), points[i].y.as_fraction()
                xj, yj = points[j].x.as_fraction(), points[j].y.as_fraction()
                dx, dy = xj - xi, yj - yi
                cross = dx * (anchor.y.as_fraction() - yi) - dy * (
                    anchor.x.as_fraction() - xi
                )
                expected = (cross * cross) / (dx * dx + dy * dy)
                assert entry.squared_distance.as_fraction() == expected

    def test_result_rejects_forged_squared_distance(self):
        request = _request(
            _point("0", "0"),
            (_point("1", "0"), _point("0", "1")),
        )
        result = compute_pinned_distances(request)
        payload = result.model_dump()
        payload["lines"][0]["squared_distance"] = {"num": "7", "den": "1"}
        with pytest.raises(ValidationError, match="pinned distance"):
            PinnedDistanceResult.model_validate(payload)

    def test_result_rejects_unreduced_squared_distance(self):
        request = _request(
            _point("0", "0"),
            (_point("1", "0"), _point("0", "1")),
        )
        payload = compute_pinned_distances(request).model_dump()
        payload["lines"][0]["squared_distance"] = {"num": "2", "den": "2"}
        with pytest.raises(ValidationError):
            PinnedDistanceResult.model_validate(payload)

    def test_oversized_coordinate_rejected_at_admission(self):
        big = "1" * 257
        with pytest.raises(ValidationError, match="digit bound"):
            _request(_point(big, "0"), (_point("0", "0"), _point("1", "0")))

    def test_duplicate_points_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            _request(
                _point("0", "0"),
                (_point("1", "0"), _point("1", "0")),
            )

    def test_line_distance_entry_requires_canonical_value(self):
        with pytest.raises(ValidationError):
            LineDistanceEntry(
                squared_distance={"num": "1", "den": "0"},
                source_pairs=((0, 1),),
            )
