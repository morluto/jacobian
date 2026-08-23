"""Source-binding tests for the pinned-distance profile."""

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
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
            entry.squared_distance.as_fraction() for entry in result.lines
        )
        assert distances[0] == 0

    def test_minimum_entry_is_ledger_member(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        assert result.min_squared_distance in result.lines

    def test_distances_are_canonical_rationals(self):
        """Every entry distance is the domain-owned canonical rational value."""
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        assert all(
            isinstance(entry.squared_distance, CanonicalRational)
            for entry in result.lines
        )


class TestAdmissionBound:
    def test_distance_beyond_the_canonical_limit_rejected(self):
        n = 10**32767 + 1
        m = 10**32767 + 2
        with pytest.raises(ValidationError, match="canonical"):
            PinnedDistanceRequest(
                anchor=_point("0", "1", "0", "1"),
                points=[
                    _point("1", format_canonical_integer(n), "0", "1"),
                    _point("0", "1", "1", format_canonical_integer(m)),
                ],
            )

    def test_moderate_configurations_remain_admitted(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        assert PinnedDistanceResult.model_validate(result.model_dump()) == result


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
            key=lambda entry: (
                int(entry["squared_distance"]["num"])
                / int(entry["squared_distance"]["den"])
            ),
        )
        payload["min_squared_distance"] = dict(max_entry)
        with pytest.raises(ValidationError, match="minimum ledger entry"):
            PinnedDistanceResult.model_validate(payload)

    def test_tampered_line_distance_rejected(self):
        request = PinnedDistanceRequest.model_validate(UNIT_SQUARE_REQUEST)
        result = compute_pinned_distances(request)
        payload = result.model_dump()
        payload["lines"][0]["squared_distance"] = {"num": "7", "den": "3"}
        with pytest.raises(ValidationError, match="recomputed ledger"):
            PinnedDistanceResult.model_validate(payload)
