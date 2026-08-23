"""Contract tests for the pinned-distance profile operation."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import RationalPoint2D
from jacobian.math.geometry._pinned_distances import (
    PINNED_DISTANCE_OPERATIONS,
    LineDistanceEntry,
    PinnedDistanceRequest,
    PinnedDistanceResult,
    compute_pinned_distances,
)
from jacobian.math.geometry._tools import TOOLS


def _pt(x: str, y: str) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational(num=x, den="1"),
        y=CanonicalRational(num=y, den="1"),
    )


class TestRegistration:
    def test_declaration_is_reachable_from_the_geometry_tools(self) -> None:
        assert "geometry.points.compute.pinned_distances" in {
            tool.operation_id for tool in TOOLS
        }

    def test_declaration_has_an_admission_decision(self) -> None:
        from jacobian.math.geometry._admission import ADMISSIONS

        assert any(
            admission.operation_id == "geometry.points.compute.pinned_distances"
            for admission in ADMISSIONS
        )

    def test_example_runs(self) -> None:
        tool = PINNED_DISTANCE_OPERATIONS[0]
        request = tool.request_type.model_validate(tool.examples[0].input)
        result = tool.run(request)
        assert result.distinct_line_count == len(result.lines)


class TestSourceReplayBinding:
    def _request(self) -> PinnedDistanceRequest:
        return PinnedDistanceRequest(
            anchor=_pt("0", "0"),
            points=(_pt("0", "0"), _pt("3", "0"), _pt("0", "4")),
        )

    def test_known_answer(self) -> None:
        # Anchor at a configuration point: the two lines through it have
        # distance zero; the hypotenuse line has distance (3*4/5)^2 = 144/25.
        result = compute_pinned_distances(self._request())
        distances = sorted(
            entry.squared_distance.as_fraction() for entry in result.lines
        )
        assert distances == [Fraction(0), Fraction(0), Fraction(144, 25)]

    def test_pair_partition_matches_generating_pairs(self) -> None:
        result = compute_pinned_distances(self._request())
        by_pairs = {entry.source_pairs: entry for entry in result.lines}
        assert by_pairs[((0, 1),)].squared_distance.as_fraction() == 0
        assert by_pairs[((0, 2),)].squared_distance.as_fraction() == 0
        hypotenuse = by_pairs[((1, 2),)]
        assert hypotenuse.squared_distance.as_fraction() == Fraction(144, 25)

    def test_collinear_configuration_single_line(self) -> None:
        result = compute_pinned_distances(
            PinnedDistanceRequest(
                anchor=_pt("0", "0"),
                points=(_pt("1", "0"), _pt("2", "0"), _pt("4", "0")),
            )
        )
        assert result.distinct_line_count == 1
        entry = result.lines[0]
        assert entry.squared_distance.as_fraction() == 0
        assert entry.source_pairs == ((0, 1), (0, 2), (1, 2))

    def test_rational_coordinates_exact_distance(self) -> None:
        result = compute_pinned_distances(
            PinnedDistanceRequest(
                anchor=_pt("0", "0"),
                points=(_pt("1", "0"), _pt("0", "1")),
            )
        )
        assert result.distinct_line_count == 1
        # Line x + y = 1 has squared distance 1/2 from the origin.
        assert result.lines[0].squared_distance.as_fraction() == Fraction(1, 2)

    def test_round_trip_replay(self) -> None:
        result = compute_pinned_distances(self._request())
        assert PinnedDistanceResult.model_validate(result.model_dump()) == result

    def test_empty_profile_with_distinct_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="replay"):
            PinnedDistanceResult(
                anchor=_pt("0", "0"),
                points=(_pt("0", "0"), _pt("1", "0")),
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )

    def test_subcardinal_result_rejected(self) -> None:
        # A relayed result must satisfy the request's minimum cardinality:
        # an empty or single-point profile is never a complete answer.
        for points in ((), (_pt("0", "0"),)):
            with pytest.raises(ValidationError, match="between 2 and"):
                PinnedDistanceResult(
                    anchor=_pt("0", "0"),
                    points=points,
                    lines=(),
                    distinct_line_count=0,
                    min_squared_distance=None,
                )

    def test_forged_entry_rejected(self) -> None:
        genuine = compute_pinned_distances(self._request())
        forged_entry = LineDistanceEntry(
            squared_distance=CanonicalRational.from_fraction(Fraction(7, 1)),
            source_pairs=genuine.lines[0].source_pairs,
        )
        with pytest.raises(ValidationError, match="replay"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=(forged_entry,),
                distinct_line_count=1,
                min_squared_distance=forged_entry,
            )

    def test_forged_source_pair_partition_rejected(self) -> None:
        genuine = compute_pinned_distances(self._request())
        payload = genuine.model_dump()
        payload["lines"][0]["source_pairs"] = ((0, 1), (1, 2))
        with pytest.raises(ValidationError, match="replay"):
            PinnedDistanceResult.model_validate(payload)

    def test_forged_line_count_rejected(self) -> None:
        genuine = compute_pinned_distances(self._request())
        payload = genuine.model_dump()
        payload["distinct_line_count"] = len(payload["lines"]) + 1
        with pytest.raises(ValidationError, match="line count"):
            PinnedDistanceResult.model_validate(payload)

    def test_detached_minimum_rejected(self) -> None:
        genuine = compute_pinned_distances(self._request())
        other = next(e for e in genuine.lines if e != genuine.min_squared_distance)
        with pytest.raises(ValidationError, match="minimum"):
            PinnedDistanceResult(
                anchor=genuine.anchor,
                points=genuine.points,
                lines=genuine.lines,
                distinct_line_count=genuine.distinct_line_count,
                min_squared_distance=other,
            )

    def test_duplicate_request_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            PinnedDistanceRequest(
                anchor=_pt("0", "0"),
                points=(_pt("1", "0"), _pt("1", "0")),
            )


class TestAdmissionBudget:
    def test_huge_coordinates_rejected(self) -> None:
        big = "1" + "0" * 256
        with pytest.raises(ValidationError):
            PinnedDistanceRequest(
                anchor=_pt("0", "0"),
                points=(
                    _pt("1", "0"),
                    RationalPoint2D(
                        x=CanonicalRational(num="1", den=big),
                        y=CanonicalRational(num="3", den=big),
                    ),
                ),
            )

    def test_point_count_capped_by_enumeration_budget(self) -> None:
        points = tuple(_pt(str(index), str(index * index)) for index in range(33))
        with pytest.raises(ValidationError):
            PinnedDistanceRequest(anchor=_pt("0", "0"), points=points)


class TestResultAdmissionBounds:
    def test_oversized_result_point_set_rejected_before_replay(self) -> None:
        """A directly validated result cannot bypass the cardinality cap."""
        points = tuple(_pt(str(index), str(index * index)) for index in range(40))
        with pytest.raises(ValidationError, match="between 2 and"):
            PinnedDistanceResult(
                anchor=_pt("0", "0"),
                points=points,
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )

    def test_duplicate_points_in_result_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            PinnedDistanceResult(
                anchor=_pt("0", "0"),
                points=(_pt("1", "0"), _pt("1", "0")),
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )
