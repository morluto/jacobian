"""Exact geometry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.exact._models import (
    DistanceGraphRequest,
    DistanceProfileRequest,
    DistanceProfileResult,
    PinnedLineDistanceRequest,
    PinnedLineDistanceResult,
)
from jacobian.math.geometry.exact.operations import (
    distance_graph,
    distance_profile,
    pinned_line_distance_profile,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _run_distance_profile(request: DistanceProfileRequest) -> DistanceProfileResult:
    return distance_profile(request.configuration)


def _run_distance_graph(request: DistanceGraphRequest) -> IndexedSimpleUndirectedGraph:
    return distance_graph(request.configuration, request.target_squared_distance)


def _run_pinned_line_distance_profile(
    request: PinnedLineDistanceRequest,
) -> PinnedLineDistanceResult:
    return pinned_line_distance_profile(request.configuration, request.anchor)


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


INVERTED_ORTHOCENTRIC = {
    "configuration": {
        "points": [
            {
                "label": "b",
                "coordinates": [{"num": "1", "den": "4"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "c",
                "coordinates": [{"num": "1", "den": "5"}, {"num": "2", "den": "5"}],
            },
            {
                "label": "h",
                "coordinates": [{"num": "4", "den": "13"}, {"num": "6", "den": "13"}],
            },
        ]
    },
    "anchor": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
}
UNIT_SQUARE_ORIGIN = {
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
    },
    "anchor": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "geometry.points.distance_profile.compute",
        "Compute pairwise distance profile",
        "Given a finite set of labelled rational points, compute the exact "
        "squared distance for every unordered pair and return the distance "
        "multiplicity profile.",
        DistanceProfileRequest,
        DistanceProfileResult,
        _run_distance_profile,
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
        "the canonical integer-indexed simple graph whose edges connect pairs "
        "at exactly that distance.",
        DistanceGraphRequest,
        IndexedSimpleUndirectedGraph,
        _run_distance_graph,
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
        "geometry.points.pinned_line_distance_profile.compute",
        "Compute pinned distances to pair-spanned lines",
        "Given a bounded labelled rational planar point configuration with "
        "distinct coordinates (no two points share the same location) and a "
        "planar rational anchor (both at most 256 digits per coordinate), "
        "construct every distinct line spanned by a pair of configuration "
        "points, compute the exact squared distance from the anchor to each "
        "line, collapse pairs defining the same geometric line while retaining "
        "every source pair, and group lines at equal squared distance into a "
        "sorted multiplicity partition.",
        PinnedLineDistanceRequest,
        PinnedLineDistanceResult,
        _run_pinned_line_distance_profile,
        "geometry",
        "pinned-distance",
        "lines",
        "exact",
        examples=(
            example(
                "inverted_orthocentric_equal_distance",
                (
                    "For B'=(1/4,0), C'=(1/5,2/5), H'=(4/13,6/13) with anchor "
                    "(0,0), all three pair-spanned lines have exact squared "
                    "distance 4/65; the configuration must be planar with "
                    "distinct coordinates and a planar anchor."
                ),
                INVERTED_ORTHOCENTRIC,
            ),
            example(
                "unit_square_anchor_origin",
                (
                    "Unit square with anchor at the origin; opposite sides and "
                    "diagonals give distinct pinned distances. The configuration "
                    "must have distinct coordinates and the anchor must be planar."
                ),
                UNIT_SQUARE_ORIGIN,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
