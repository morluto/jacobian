"""Known-answer, adversarial, and boundary tests for pinned distances."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.geometry._pinned_distances import (
    LineDistanceEntry,
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)


def _point(nx: str, ny: str) -> dict:
    return {
        "x": {"num": nx, "den": "1"},
        "y": {"num": ny, "den": "1"},
    }


def _request(anchor: dict, points: list[dict]) -> PinnedDistanceRequest:
    return PinnedDistanceRequest.model_validate({"anchor": anchor, "points": points})


_UNIT_SQUARE_ANCHOR = _point("0", "0")
_UNIT_SQUARE_POINTS = [
    _point("0", "0"),
    _point("1", "0"),
    _point("1", "1"),
    _point("0", "1"),
]


class TestPinnedDistancesKnownAnswer:
    def test_unit_square_profile(self) -> None:
        result = compute_pinned_distances(
            _request(_UNIT_SQUARE_ANCHOR, _UNIT_SQUARE_POINTS)
        )
        assert result.complete is True
        assert result.distinct_line_count == 6

        zero_entries = [e for e in result.lines if e.squared_distance_numerator == "0"]
        assert [e.source_pairs for e in zero_entries] == [
            ((0, 1),),
            ((0, 2),),
            ((0, 3),),
        ]
        half_entries = [
            e
            for e in result.lines
            if (e.squared_distance_numerator, e.squared_distance_denominator)
            == ("1", "2")
        ]
        assert len(half_entries) == 1
        assert half_entries[0].source_pairs == ((1, 3),)
        one_entries = [
            e
            for e in result.lines
            if e.squared_distance_numerator == "1"
            and e.squared_distance_denominator == "1"
        ]
        assert {e.source_pairs for e in one_entries} == {((1, 2),), ((2, 3),)}

        assert result.min_squared_distance is not None
        assert result.min_squared_distance.squared_distance_numerator == "0"
        assert Fraction(
            int(result.min_squared_distance.squared_distance_numerator),
            int(result.min_squared_distance.squared_distance_denominator),
        ) == min(
            Fraction(
                int(e.squared_distance_numerator), int(e.squared_distance_denominator)
            )
            for e in result.lines
        )

    def test_result_round_trips_through_validation(self) -> None:
        result = compute_pinned_distances(
            _request(_UNIT_SQUARE_ANCHOR, _UNIT_SQUARE_POINTS)
        )
        replayed = PinnedDistanceResult.model_validate(result.model_dump())
        assert replayed == result


class TestPinnedDistancesSourceReplay:
    def _genuine(self) -> PinnedDistanceResult:
        return compute_pinned_distances(
            _request(_UNIT_SQUARE_ANCHOR, _UNIT_SQUARE_POINTS)
        )

    def test_arbitrary_entry_with_empty_ledger_rejected(self) -> None:
        genuine = self._genuine()
        forged = tuple(
            LineDistanceEntry(
                squared_distance_numerator="123456789",
                squared_distance_denominator="7",
                source_pairs=(),
            )
            for _ in genuine.lines
        )
        with pytest.raises(ValidationError, match="exact profile"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=forged,
                distinct_line_count=len(forged),
                min_squared_distance=forged[0],
            )

    def test_tampered_distance_on_retained_pair_rejected(self) -> None:
        genuine = self._genuine()
        tampered = tuple(
            LineDistanceEntry(
                squared_distance_numerator=e.squared_distance_numerator + "0",
                squared_distance_denominator=e.squared_distance_denominator,
                source_pairs=e.source_pairs,
            )
            for e in genuine.lines
        )
        with pytest.raises(ValidationError, match="exact profile"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=tampered,
                distinct_line_count=len(tampered),
                min_squared_distance=tampered[0],
            )

    def test_dropped_source_pair_rejected(self) -> None:
        anchor = _point("0", "1")
        points = [
            _point("0", "0"),
            _point("1", "0"),
            _point("2", "0"),
            _point("3", "0"),
        ]
        genuine = compute_pinned_distances(_request(anchor, points))
        assert len(genuine.lines) == 1
        full = genuine.lines[0]
        assert len(full.source_pairs) == 6
        trimmed = LineDistanceEntry(
            squared_distance_numerator=full.squared_distance_numerator,
            squared_distance_denominator=full.squared_distance_denominator,
            source_pairs=full.source_pairs[:1],
        )
        with pytest.raises(ValidationError, match="exact profile"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=[trimmed],
                distinct_line_count=1,
                min_squared_distance=trimmed,
            )

    def test_wrong_count_rejected(self) -> None:
        genuine = self._genuine()
        with pytest.raises(ValidationError, match="line count"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=genuine.lines[:-1],
                distinct_line_count=len(genuine.lines) - 1,
                min_squared_distance=None,
            )


class TestPinnedDistancesBoundaries:
    def test_duplicate_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            _request(
                _point("0", "0"),
                [_point("1", "1"), _point("1", "1")],
            )

    def test_large_coordinates_avoid_integer_string_limit(self) -> None:
        """2048-digit coordinates yield ~8000-digit exact numerators."""
        huge_a = "9" * 2048
        huge_b = "8" * 2048
        result = compute_pinned_distances(
            _request(
                _point(huge_a, huge_b),
                [_point(huge_a, huge_a), _point(huge_b, huge_b), _point("0", "1")],
            )
        )
        max_digits = max(
            len(e.squared_distance_numerator) + len(e.squared_distance_denominator)
            for e in result.lines
        )
        assert max_digits > 4300
        # Replay validation inside the result model already re-derived every
        # entry; a second construction must succeed identically.
        replayed = PinnedDistanceResult.model_validate(result.model_dump())
        assert replayed == result
