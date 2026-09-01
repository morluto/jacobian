"""Projective coordinate operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.projective.coordinates._models import (
    ChartTransitionRequest,
    ChartTransitionResult,
    RationalPointConstructRequest,
    RationalPointConstructResult,
    StandardChartRequest,
    StandardChartResult,
)
from jacobian.math.geometry.projective.coordinates.operations import (
    chart_transition,
    rational_projective_point,
    standard_chart,
)


def _construct(request: RationalPointConstructRequest) -> RationalPointConstructResult:
    return rational_projective_point(request.coordinates)


def _standard_chart(request: StandardChartRequest) -> StandardChartResult:
    return standard_chart(request.point, request.chart_index)


def _chart_transition(request: ChartTransitionRequest) -> ChartTransitionResult:
    return chart_transition(request.point, request.chart_i, request.chart_j)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="projective.rational_point.construct",
        title="Construct a canonical rational projective point",
        description="Canonicalize a rational projective point by scaling so the first "
        "nonzero coordinate is 1.",
        request_type=RationalPointConstructRequest,
        result_type=RationalPointConstructResult,
        run=_construct,
        tags=("projective", "rational", "exact"),
        examples=(
            OperationExample(
                name="p1_point",
                description="Construct [2 : 4] in P^1(Q).",
                input={
                    "coordinates": [
                        {"num": "2", "den": "1"},
                        {"num": "4", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="projective.standard_chart.compute",
        title="Dehomogenize at a standard affine chart",
        description="Dehomogenize a projective point at the given chart index by "
        "dividing all coordinates by that coordinate.",
        request_type=StandardChartRequest,
        result_type=StandardChartResult,
        run=_standard_chart,
        tags=("projective", "affine-chart", "exact"),
        examples=(
            OperationExample(
                name="chart_0",
                description="Dehomogenize [1 : 2 : 3] at chart 0.",
                input={
                    "point": {
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                    "chart_index": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="projective.chart_transition.compute",
        title="Compute the transition map between two charts",
        description="Return the complete target-chart coordinates for a projective point, "
        "or OUTSIDE_TARGET_CHART when the target coordinate vanishes.",
        request_type=ChartTransitionRequest,
        result_type=ChartTransitionResult,
        run=_chart_transition,
        tags=("projective", "chart-transition", "exact"),
        examples=(
            OperationExample(
                name="transition_0_to_1",
                description="Transition from chart 0 to chart 1 for [1 : 2 : 3].",
                input={
                    "point": {
                        "coordinates": [
                            {"num": "1", "den": "1"},
                            {"num": "2", "den": "1"},
                            {"num": "3", "den": "1"},
                        ],
                    },
                    "chart_i": 0,
                    "chart_j": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
