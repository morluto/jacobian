"""Tests for projective coordinate operations."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.projective_coords_ops._models import (
    ChartTransitionRequest,
    ChartTransitionResult,
    RationalPointConstructRequest,
    RationalProjectivePoint,
    StandardChartRequest,
)
from jacobian.math.projective_coords_ops._operations import (
    compute_chart_transition,
    compute_rational_point_construct,
    compute_standard_chart,
)
from jacobian.math.projective_coords_ops._tools import TOOLS


def _r(num: str, den: str = "1") -> CanonicalRational:
    return CanonicalRational(num=num, den=den)


def _point(*coordinates: CanonicalRational) -> RationalProjectivePoint:
    return RationalProjectivePoint(coordinates=coordinates)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "projective.rational_point.construct",
        "projective.standard_chart.compute",
        "projective.chart_transition.compute",
    }


def test_rational_point_construct() -> None:
    request = RationalPointConstructRequest(coordinates=(_r("2"), _r("4")))
    result = compute_rational_point_construct(request)
    assert result.canonical[0].as_fraction() == 1
    assert result.canonical[1].as_fraction() == 2


def test_standard_chart() -> None:
    request = StandardChartRequest(
        point=_point(_r("1"), _r("2"), _r("3")),
        chart_index=0,
    )
    result = compute_standard_chart(request)
    assert result.affine_point[0].as_fraction() == 2
    assert result.affine_point[1].as_fraction() == 3


def test_chart_transition() -> None:
    request = ChartTransitionRequest(
        point=_point(_r("1"), _r("2"), _r("3")),
        chart_i=0,
        chart_j=1,
    )
    result = compute_chart_transition(request)
    assert result.status == "DEFINED"
    assert result.transition is not None
    assert tuple(value.as_fraction() for value in result.transition) == (
        Fraction(1, 2),
        Fraction(3, 2),
    )


def test_chart_transition_is_invariant_under_homogeneous_rescaling() -> None:
    original = ChartTransitionRequest(
        point=_point(_r("1"), _r("2"), _r("3")),
        chart_i=0,
        chart_j=1,
    )
    rescaled = ChartTransitionRequest(
        point=_point(_r("5"), _r("10"), _r("15")),
        chart_i=0,
        chart_j=1,
    )

    assert (
        compute_chart_transition(original).transition
        == compute_chart_transition(rescaled).transition
    )


def test_chart_transition_reports_outside_target_chart() -> None:
    result = compute_chart_transition(
        ChartTransitionRequest(
            point=_point(_r("1"), _r("0"), _r("3")),
            chart_i=0,
            chart_j=1,
        )
    )

    assert result.status == "OUTSIDE_TARGET_CHART"
    assert result.transition is None


def test_chart_transition_rejects_unrepresentable_ratio_growth() -> None:
    component = "1" + "0" * 16_384

    with pytest.raises(ValidationError, match="ratio budget"):
        ChartTransitionRequest(
            point=RationalProjectivePoint(
                coordinates=(
                    CanonicalRational(num=component, den="1"),
                    CanonicalRational(num="1", den=component),
                )
            ),
            chart_i=0,
            chart_j=1,
        )


def test_chart_transition_round_trips_between_defined_charts() -> None:
    point = _point(_r("2"), _r("3"), _r("5"))
    forward = compute_chart_transition(
        ChartTransitionRequest(point=point, chart_i=0, chart_j=1)
    )
    backward = compute_chart_transition(
        ChartTransitionRequest(point=point, chart_i=1, chart_j=0)
    )

    assert forward.transition == (_r("2", "3"), _r("5", "3"))
    assert backward.transition == (_r("3", "2"), _r("5", "2"))


def test_chart_transition_result_rejects_an_incomplete_target_chart() -> None:
    with pytest.raises(ValidationError, match="every target-chart coordinate"):
        ChartTransitionResult(
            status="DEFINED",
            transition=(_r("1"),),
            chart_i=0,
            chart_j=1,
            projective_dimension=2,
        )
