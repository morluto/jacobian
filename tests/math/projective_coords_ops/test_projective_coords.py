"""Tests for projective coordinate operations."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.projective_coords_ops._models import (
    ChartTransitionRequest,
    RationalPointConstructRequest,
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
        point={
            "coordinates": [_r("1"), _r("2"), _r("3")],
        },
        chart_index=0,
    )
    result = compute_standard_chart(request)
    assert result.affine_point[0].as_fraction() == 2
    assert result.affine_point[1].as_fraction() == 3


def test_chart_transition() -> None:
    request = ChartTransitionRequest(
        point={
            "coordinates": [_r("1"), _r("2"), _r("3")],
        },
        chart_i=0,
        chart_j=1,
    )
    result = compute_chart_transition(request)
    assert result.transition[0].as_fraction() == Fraction(3, 2)
