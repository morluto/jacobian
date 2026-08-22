"""Contract tests for the pinned-distance profile operation."""

from __future__ import annotations

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
            (
                entry.squared_distance_numerator,
                entry.squared_distance_denominator,
            )
            for entry in result.lines
        )
        assert distances == [("0", "1"), ("0", "1"), ("144", "25")]

    def test_empty_profile_with_distinct_points_rejected(self) -> None:
        with pytest.raises(ValidationError, match="replay"):
            PinnedDistanceResult(
                anchor=_pt("0", "0"),
                points=(_pt("0", "0"), _pt("1", "0")),
                lines=(),
                distinct_line_count=0,
                min_squared_distance=None,
            )

    def test_forged_entry_rejected(self) -> None:
        genuine = compute_pinned_distances(self._request())
        forged_entry = LineDistanceEntry(
            squared_distance_numerator="7",
            squared_distance_denominator="1",
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
        points = tuple(
            _pt(str(index), str(index * index))
            for index in range(33)
        )
        with pytest.raises(ValidationError):
            PinnedDistanceRequest(anchor=_pt("0", "0"), points=points)
