"""Projective coordinate operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.projective.coordinates._models import (
    ChartTransitionRequest,
    ChartTransitionResult,
    RationalPointConstructRequest,
    RationalPointConstructResult,
    StandardChartRequest,
    StandardChartResult,
)
from jacobian.math.geometry.projective.coordinates._operations import (
    compute_chart_transition,
    compute_rational_point_construct,
    compute_standard_chart,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "projective.rational_point.construct",
        "Construct a canonical rational projective point",
        "Canonicalize a rational projective point by scaling so the first "
        "nonzero coordinate is 1.",
        RationalPointConstructRequest,
        RationalPointConstructResult,
        compute_rational_point_construct,
        "projective",
        "rational",
        "exact",
        examples=(
            example(
                "p1_point",
                "Construct [2 : 4] in P^1(Q).",
                {
                    "coordinates": [
                        {"num": "2", "den": "1"},
                        {"num": "4", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    _op(
        "projective.standard_chart.compute",
        "Dehomogenize at a standard affine chart",
        "Dehomogenize a projective point at the given chart index by "
        "dividing all coordinates by that coordinate.",
        StandardChartRequest,
        StandardChartResult,
        compute_standard_chart,
        "projective",
        "affine-chart",
        "exact",
        examples=(
            example(
                "chart_0",
                "Dehomogenize [1 : 2 : 3] at chart 0.",
                {
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
    _op(
        "projective.chart_transition.compute",
        "Compute the transition map between two charts",
        "Return the complete target-chart coordinates for a projective point, "
        "or OUTSIDE_TARGET_CHART when the target coordinate vanishes.",
        ChartTransitionRequest,
        ChartTransitionResult,
        compute_chart_transition,
        "projective",
        "chart-transition",
        "exact",
        examples=(
            example(
                "transition_0_to_1",
                "Transition from chart 0 to chart 1 for [1 : 2 : 3].",
                {
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
