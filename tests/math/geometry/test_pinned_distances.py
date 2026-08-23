"""Contract tests for the bounded exact pinned-distance profile operation."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._pinned_distances import (
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x={"num": x, "den": "1"},
        y={"num": y, "den": "1"},
    )


class TestPinnedDistancesKnownAnswer:
    def test_unit_distance_to_baseline(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point("0", "1"),
            points=(_point("0", "0"), _point("2", "0")),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 1
        entry = result.lines[0]
        assert entry.squared_distance_numerator == "1"
        assert entry.squared_distance_denominator == "1"
        assert entry.source_pairs == ((0, 1),)
        assert result.min_squared_distance is not None
        assert result.min_squared_distance.squared_distance_numerator == "1"

    def test_three_lines_from_four_points(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point("0", "0"),
            points=(_point("1", "0"), _point("0", "1"), _point("1", "1")),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 3
        squared = {
            (
                entry.squared_distance_numerator,
                entry.squared_distance_denominator,
            )
            for entry in result.lines
        }
        assert ("1", "2") in squared
        assert sum(len(entry.source_pairs) for entry in result.lines) == 3
        assert result.min_squared_distance.squared_distance_numerator == "1"


class TestPinnedDistanceAdmissionAndBinding:
    def test_extreme_coordinate_height_is_rejected_at_admission(self) -> None:
        """The thread example must fail at admission, not inside execution.

        Anchor (0, 10^4095) with points (0,0) and (1,0) has a squared-distance
        numerator near 10^8190; the previous implementation raised CPython's
        int-to-string digit-limit ValueError after accepting the request.
        """
        with pytest.raises(ValidationError, match="32-digit"):
            PinnedDistanceRequest(
                anchor=_point("0", "1" + "0" * 4095),
                points=(_point("0", "0"), _point("1", "0")),
            )

    def test_largest_admitted_coordinates_round_trip(self) -> None:
        """A 32-digit configuration computes and revalidates its profile."""
        tall = "1" + "0" * 30 + "7"
        request = PinnedDistanceRequest(
            anchor=_point(tall, tall),
            points=(_point("0", "0"), _point("1", "-" + tall)),
        )
        result = compute_pinned_distances(request)
        assert result.distinct_line_count >= 1
        PinnedDistanceResult.model_validate(result.model_dump(mode="json"))

    def test_forged_distance_is_rejected_by_replay(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point("0", "1"),
            points=(_point("0", "0"), _point("2", "0")),
        )
        result = compute_pinned_distances(request)
        payload = result.model_dump()
        payload["lines"][0]["squared_distance_numerator"] = "5"
        with pytest.raises(ValidationError):
            PinnedDistanceResult.model_validate(payload)

    def test_duplicate_points_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            PinnedDistanceRequest(
                anchor=_point("0", "0"),
                points=(_point("1", "0"), _point("1", "0")),
            )
