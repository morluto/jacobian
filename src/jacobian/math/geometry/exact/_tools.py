"""Exact geometry operation declarations."""

from typing import Any

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
    MathTool(
        operation_id="geometry.points.distance_profile.compute",
        title="Compute pairwise distance profile",
        description="Given a finite set of labelled rational points, compute the exact "
        "squared distance for every unordered pair and return the distance "
        "multiplicity profile.",
        request_type=DistanceProfileRequest,
        result_type=DistanceProfileResult,
        run=_run_distance_profile,
        tags=("geometry", "distance", "exact"),
        examples=(
            OperationExample(
                name="unit_square_profile",
                description="Distance profile of the unit square.",
                input=UNIT_SQUARE,
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.points.distance_graph.compute",
        title="Build distance-selected graph",
        description="Given a point configuration and a target squared distance, return "
        "the canonical integer-indexed simple graph whose edges connect pairs "
        "at exactly that distance.",
        request_type=DistanceGraphRequest,
        result_type=IndexedSimpleUndirectedGraph,
        run=_run_distance_graph,
        tags=("geometry", "distance-graph", "exact"),
        examples=(
            OperationExample(
                name="unit_square_distance_1",
                description="Graph of unit-distance pairs in the unit square.",
                input={
                    **UNIT_SQUARE,
                    "target_squared_distance": {"num": "1", "den": "1"},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.points.pinned_line_distance_profile.compute",
        title="Compute pinned distances to pair-spanned lines",
        description="Given a bounded labelled rational planar point configuration with "
        "distinct coordinates (no two points share the same location) and a "
        "planar rational anchor (both at most 256 digits per coordinate), "
        "construct every distinct line spanned by a pair of configuration "
        "points, compute the exact squared distance from the anchor to each "
        "line, collapse pairs defining the same geometric line while retaining "
        "every source pair, and group lines at equal squared distance into a "
        "sorted multiplicity partition.",
        request_type=PinnedLineDistanceRequest,
        result_type=PinnedLineDistanceResult,
        run=_run_pinned_line_distance_profile,
        tags=("geometry", "pinned-distance", "lines", "exact"),
        examples=(
            OperationExample(
                name="inverted_orthocentric_equal_distance",
                description=(
                    "For B'=(1/4,0), C'=(1/5,2/5), H'=(4/13,6/13) with anchor "
                    "(0,0), all three pair-spanned lines have exact squared "
                    "distance 4/65; the configuration must be planar with "
                    "distinct coordinates and a planar anchor."
                ),
                input=INVERTED_ORTHOCENTRIC,
            ),
            OperationExample(
                name="unit_square_anchor_origin",
                description=(
                    "Unit square with anchor at the origin; opposite sides and "
                    "diagonals give distinct pinned distances. The configuration "
                    "must have distinct coordinates and the anchor must be planar."
                ),
                input=UNIT_SQUARE_ORIGIN,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
