"""Source-binding tests for the pinned-distance profile."""

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._pinned_distances import (
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(num_x, den_x, num_y, den_y):
    return {
        "x": {"num": num_x, "den": den_x},
        "y": {"num": num_y, "den": den_y},
    }


UNIT_SQUARE_REQUEST = {
    "anchor": _point("0", "1", "0", "1"),
    "points": [
        _point("0", "1", "0", "1"),
        _point("1", "1", "0", "1"),
        _point("1", "1", "1", "1"),
        _point("0", "1", "1", "1"),
    ],
}


class TestKnownAnswers:
    def test_unit_square_ledger(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        assert result.distinct_line_count == 6
        assert len(result.lines) == 6
        distances = sorted(
            int(entry.squared_distance_numerator) / int(entry.squared_distance_denominator)
            for entry in result.lines
        )
        assert distances[0] == 0

    def test_minimum_entry_is_ledger_member(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        assert result.min_squared_distance in result.lines


class TestResultBinding:
    def test_tampered_minimum_source_pairs_rejected(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        payload = result.model_dump()
        payload["min_squared_distance"]["source_pairs"] = ((99, 100),)
        with pytest.raises(ValidationError, match="minimum ledger entry"):
            PinnedDistanceResult.model_validate(payload)

    def test_swapped_minimum_distance_rejected(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        payload = result.model_dump()
        max_entry = max(
            payload["lines"],
            key=lambda entry: int(entry["squared_distance_numerator"])
            / int(entry["squared_distance_denominator"]),
        )
        payload["min_squared_distance"] = dict(max_entry)
        with pytest.raises(ValidationError, match="minimum ledger entry"):
            PinnedDistanceResult.model_validate(payload)
