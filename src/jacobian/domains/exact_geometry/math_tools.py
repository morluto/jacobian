"""Exact geometry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact_geometry import (
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceProfileRequest,
    DistanceProfileResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.exact_geometry.operations import (
    compute_distance_graph,
    compute_distance_profile,
)
from jacobian.math_tools import MathTool


def _op[
    RequestT: ContractModel,
    ResultT: ContractModel,
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
)


__all__ = ["EXACT_GEOMETRY_OPERATIONS"]
