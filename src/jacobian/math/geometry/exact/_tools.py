"""Exact geometry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.exact._models import (
    CollinearTriplesRequest,
    ConcyclicQuadruplesRequest,
    DistanceGraphRequest,
    DistanceGraphResult,
    DistanceProfileRequest,
    DistanceProfileResult,
    IncidenceSearchResult,
    PinnedLineDistanceRequest,
    PinnedLineDistanceResult,
)
from jacobian.math.geometry.exact._operations import (
    compute_collinear_triples,
    compute_concyclic_quadruples,
    compute_distance_graph,
    compute_distance_profile,
    compute_pinned_line_distance_profile,
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


NO_COLLINEAR_GENERAL_POSITION = {
    "configuration": {
        "points": [
            {
                "label": "a",
                "coordinates": [{"num": "-1", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "b",
                "coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "c",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
            },
            {
                "label": "d",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "-2", "den": "1"}],
            },
        ]
    }
}
HAS_COLLINEAR_TRIPLE = {
    "configuration": {
        "points": [
            {
                "label": "a",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "b",
                "coordinates": [{"num": "2", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "c",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
            },
            {
                "label": "d",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "-2", "den": "1"}],
            },
        ]
    }
}
HAS_CONCYCLIC_QUADRUPLE = {
    "configuration": {
        "points": [
            {
                "label": "a",
                "coordinates": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "b",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            },
            {
                "label": "c",
                "coordinates": [{"num": "-1", "den": "1"}, {"num": "0", "den": "1"}],
            },
            {
                "label": "d",
                "coordinates": [{"num": "0", "den": "1"}, {"num": "-1", "den": "1"}],
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
        "geometry.points.collinear_triples.find",
        "Find collinear triples in a point configuration",
        "Given a bounded labelled rational planar point configuration, find "
        "every collinear triple of points or establish after complete bounded "
        "enumeration that none exists. Exactness rests on a vanishing 3x3 "
        "determinant over the rationals.",
        CollinearTriplesRequest,
        IncidenceSearchResult,
        compute_collinear_triples,
        "geometry",
        "incidence",
        "collinear",
        "exact",
        examples=(
            example(
                "general_position_no_collinear",
                (
                    "Four points A=(-1,0), B=(1,0), C=(0,2), D=(0,-2) have no "
                    "three collinear. The configuration must be planar with "
                    "3..40 points, pairwise distinct coordinates, each "
                    "coordinate at most 64 digits."
                ),
                NO_COLLINEAR_GENERAL_POSITION,
            ),
            example(
                "collinear_triple_present",
                (
                    "A=(0,0), B=(2,0), C=(0,2), D=(0,-2): A, C, D are collinear "
                    "on x=0, so a collinear triple exists. The configuration "
                    "must be planar with 3..40 points, pairwise distinct "
                    "coordinates, each coordinate at most 64 digits."
                ),
                HAS_COLLINEAR_TRIPLE,
            ),
        ),
    ),
    _op(
        "geometry.points.concyclic_quadruples.find",
        "Find concyclic quadruples in a point configuration",
        "Given a bounded labelled rational planar point configuration, find "
        "every concyclic quadruple of points or establish after complete "
        "bounded enumeration that none exists. Exactness rests on a vanishing "
        "4x4 determinant (the circle equation x^2+y^2+Dx+Ey+F=0) over the "
        "rationals.",
        ConcyclicQuadruplesRequest,
        IncidenceSearchResult,
        compute_concyclic_quadruples,
        "geometry",
        "incidence",
        "concyclic",
        "exact",
        examples=(
            example(
                "unit_circle_concyclic",
                (
                    "Points (1,0),(0,1),(-1,0),(0,-1) on the unit circle "
                    "are concyclic. Requires a planar configuration of "
                    "4..18 points with pairwise distinct coordinates, "
                    "each coordinate at most 64 digits, within the joint "
                    "work budget C(n,4)*h <= 65536."
                ),
                HAS_CONCYCLIC_QUADRUPLE,
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
        compute_pinned_line_distance_profile,
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


TOOLS = EXACT_GEOMETRY_OPERATIONS

__all__ = ["TOOLS"]
