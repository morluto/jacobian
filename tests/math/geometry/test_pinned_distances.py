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


class TestPinnedDistanceMinimumBinding:
    def test_forged_minimum_source_pairs_are_rejected(self) -> None:
        """The reported minimum must be the complete canonical entry."""
        request = PinnedDistanceRequest(
            anchor=_point("0", "1"),
            points=(_point("0", "0"), _point("2", "0")),
        )
        result = compute_pinned_distances(request)
        payload = result.model_dump()
        minimum = min(
            range(len(payload["lines"])),
            key=lambda i: (
                int(payload["lines"][i]["squared_distance_numerator"]),
                int(payload["lines"][i]["squared_distance_denominator"]),
            ),
        )
        payload["min_squared_distance"] = {
            **payload["lines"][minimum],
            "source_pairs": (),
        }
        with pytest.raises(ValidationError, match="selected minimum"):
            PinnedDistanceResult.model_validate(payload)


class TestPinnedDistanceResultSourceAdmission:
    def _collinear_payload(self, count: int) -> dict:
        points = [
            {
                "x": {"num": str(index), "den": "1"},
                "y": {"num": "0", "den": "1"},
            }
            for index in range(count)
        ]
        pairs = [
            [str(first), str(second)]
            for first in range(count)
            for second in range(first + 1, count)
        ]
        line = {
            "squared_distance_numerator": "0",
            "squared_distance_denominator": "1",
            "source_pairs": pairs,
        }
        return {
            "anchor": {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
            "points": points,
            "lines": [line],
            "distinct_line_count": 1,
            "min_squared_distance": line,
            "complete": True,
            "method": "EXACT_PINNED_DISTANCES",
        }

    def test_serialized_result_reapplies_the_128_point_cap(self) -> None:
        """A crafted profile cannot replay quadratic work beyond the request cap."""
        with pytest.raises(ValidationError, match="at most 128"):
            PinnedDistanceResult.model_validate(self._collinear_payload(129))

    def test_serialized_result_reapplies_coordinate_height_bounds(self) -> None:
        tall = "1" + "0" * 4096
        payload = self._collinear_payload(3)
        for point in (*payload["points"], payload["anchor"]):
            point["y"]["num"] = tall
        with pytest.raises(ValidationError, match="32-digit"):
            PinnedDistanceResult.model_validate(payload)

    def test_retained_sources_at_the_admitted_bound_still_revalidate(self) -> None:
        request = PinnedDistanceRequest(
            anchor=_point("0", "1"),
            points=(_point("0", "0"), _point("2", "0")),
        )
        result = compute_pinned_distances(request)
        PinnedDistanceResult.model_validate(result.model_dump(mode="json"))
