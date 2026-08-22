"""Exact geometry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry._models import (
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
)
from jacobian.math.geometry._operations import circumradius_profile
from jacobian.math.geometry.exact._models import (
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceProfileRequest,
    DistanceProfileResult,
)
from jacobian.math.geometry.exact._operations import (
    compute_distance_graph,
    compute_distance_profile,
)


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


UNIT_SQUARE = {
    "configuration": {
        "points": [
            {
                "label": "a",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "b",
                "coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "c",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            },
            {
                "label": "d",
                "coordinates": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
            },
        ]
    }
}


PARABOLA_COLLISION = {
    "points": [
        {
            "label": "t1",
            "point": {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
        },
        {
            "label": "t2",
            "point": {"x": {"num": "2", "den": "1"}, "y": {"num": "4", "den": "1"}},
        },
        {
            "label": "t4",
            "point": {"x": {"num": "4", "den": "1"}, "y": {"num": "16", "den": "1"}},
        },
        {
            "label": "t19",
            "point": {
                "x": {"num": "19", "den": "1"},
                "y": {"num": "361", "den": "1"},
            },
        },
        {
            "label": "t29",
            "point": {
                "x": {"num": "29", "den": "1"},
                "y": {"num": "841", "den": "1"},
            },
        },
    ]
}

EXACT_GEOMETRY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "geometry.points.distance_profile.compute",
        "Compute pairwise distance profile",
        "Given a finite set of labelled rational points, compute the exact "
        "squared distance for every unordered pair and return the distance "
        "multiplicity profile.",
        DistanceProfileRequest,
        DistanceProfileResult,
        compute_distance_profile,
        "geometry",
        "distance",
        "exact",
        examples=(
            example(
                "unit_square_profile",
                "Distance profile of the unit square.",
                UNIT_SQUARE,
            ),
        ),
    ),
    _op(
        "geometry.points.distance_graph.compute",
        "Build distance-selected graph",
        "Given a point configuration and a target squared distance, return "
        "the graph whose edges connect pairs at exactly that distance.",
        DistanceGraphRequest,
        DistanceGraphResult,
        compute_distance_graph,
        "geometry",
        "distance-graph",
        "exact",
        examples=(
            example(
                "unit_square_distance_1",
                "Graph of unit-distance pairs in the unit square.",
                {**UNIT_SQUARE, "target_squared_distance": {"num": "1", "den": "1"}},
            ),
        ),
    ),
    _op(
        "geometry.points.circumradius_profile.compute",
        "Compute circumradius profile of a point configuration",
        "Given a bounded labelled rational planar point configuration with "
        "at least three points, unique labels, and unique coordinates, "
        "compute the exact squared circumradius of every unordered triple "
        "with an explicit collinear or nondegenerate disposition, each "
        "entry replayed against the retained source configuration.",
        CircumradiusProfileRequest,
        CircumradiusProfileResult,
        circumradius_profile,
        "geometry",
        "circumradius",
        "exact",
        examples=(
            example(
                "parabola_circumradius_collision",
                (
                    "Circumradius profile of parabola points P(t)=(t,t^2) for "
                    "t in {1,2,4,19,29}; triangles (1,2,29) and (2,4,19) share "
                    "the same squared circumradius. The configuration must be "
                    "planar with at least three points."
                ),
                PARABOLA_COLLISION,
            ),
        ),
    ),
)


TOOLS = EXACT_GEOMETRY_OPERATIONS

__all__ = ["TOOLS"]
