"""Contract tests for pinned-distance to pair-spanned-lines profiles."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._pinned_distances import (
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(x: int, y: int) -> RationalPoint2D:
    return RationalPoint2D(
        x={"num": str(x), "den": "1"},
        y={"num": str(y), "den": "1"},
    )


class TestPinnedDistanceProfile:
    def test_two_points_span_one_line(self) -> None:
        """Two distinct points necessarily span exactly one line."""
        request = PinnedDistanceRequest(
            anchor=_point(0, 0),
            points=(_point(0, 1), _point(1, 0)),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 1
        assert len(result.lines) == 1
        assert result.lines[0].source_pairs == ((0, 1),)
        # Distance from (0,0) to x + y = 1 is 1/sqrt(2); squared is 1/2.
        assert result.lines[0].squared_distance_numerator == "1"
        assert result.lines[0].squared_distance_denominator == "2"

    def test_square_has_distinct_lines(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point(0, 0),
            points=(
                _point(0, 1),
                _point(1, 0),
                _point(1, 1),
            ),
        )
        result = compute_pinned_distances(request)
        # Lines x+y=1, y=1, x=1 are pairwise distinct.
        assert result.distinct_line_count == 3
        assert len(result.lines) == 3

    def test_collinear_points_merge_into_one_line(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point(0, 5),
            points=(_point(0, 0), _point(1, 0), _point(2, 0)),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 1
        assert result.lines[0].source_pairs == ((0, 1), (0, 2), (1, 2))
        assert result.min_squared_distance == result.lines[0]


class TestSourceBoundLedger:
    def test_result_replays_against_retained_sources(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point(0, 0),
            points=(_point(0, 1), _point(1, 0)),
        )
        result = compute_pinned_distances(request)
        revalidated = PinnedDistanceResult.model_validate(result.model_dump())
        assert revalidated.distinct_line_count == 1

    def test_truncated_ledger_is_rejected(self) -> None:
        """Two distinct points span one line; an empty ledger cannot validate."""
        payload = {
            "anchor": {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
            "points": [
                {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
            ],
            "lines": (),
            "distinct_line_count": 0,
            "min_squared_distance": None,
        }
        with pytest.raises(ValidationError, match="exact canonical ledger"):
            PinnedDistanceResult.model_validate(payload)

    def test_altered_distance_is_rejected(self) -> None:
        payload = {
            "anchor": {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
            "points": [
                {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
                {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
            ],
            "lines": [
                {
                    "squared_distance_numerator": "3",
                    "squared_distance_denominator": "2",
                    "source_pairs": [[0, 1]],
                }
            ],
            "distinct_line_count": 1,
            "min_squared_distance": None,
        }
        with pytest.raises(ValidationError):
            PinnedDistanceResult.model_validate(payload)

    def test_empty_ledger_cannot_come_from_retained_points(self) -> None:
        """Any two distinct retained points span a line, so an empty ledger
        with retained points is rejected as a ledger mismatch."""
        with pytest.raises(ValidationError):
            PinnedDistanceResult(
                anchor=_point(0, 0),
                points=(_point(0, 1), _point(1, 0)),
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )
