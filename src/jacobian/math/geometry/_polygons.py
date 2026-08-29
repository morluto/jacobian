"""Polygon-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.geometry._euclidean_triangulation import (
    minimum_euclidean_weight_triangulation,
)
from jacobian.math.geometry._models import (
    MAX_EUCLIDEAN_TRIANGULATION_VERTICES,
    ConvexPolygonTriangulationRequest,
    ConvexPolygonTriangulationResult,
    EuclideanConvexPolygonTriangulationRequest,
    EuclideanConvexPolygonTriangulationResult,
    PolygonPointClassificationResult,
    PolygonRequest,
    SimplePolygonDecisionResult,
    SimplePolygonPointRequest,
)
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry._tools import (
    classify_polygon_point,
    simple_polygon,
)
from jacobian.math.geometry._triangulation import minimum_weight_triangulation

_UNIT_SQUARE = [
    {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
    {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
]

POLYGON_OPERATIONS: MathTools = (
    geometry_operation(
        "geometry.polygon.triangulation.minimum_euclidean_weight.compute",
        "Compute a certified minimum Euclidean convex-polygon triangulation",
        (
            "Compute one deterministic minimum triangulation of a strict CCW convex "
            f"simple rational polygon of 4 to {MAX_EUCLIDEAN_TRIANGULATION_VERTICES} "
            "vertices, charging each selected "
            "non-hull diagonal once by its Euclidean length; admission bounds "
            "the complete serialized result - split table, echoed source "
            "ring, and metadata - against the published output bound from "
            "the exact source, so translated sources pay for their echoed "
            "coordinates even though the mathematical work depends only on "
            "pairwise coordinate differences. Returns "
            "the exact "
            "sum-of-square-roots cost expression only when every finite DP "
            "comparison is separated by a pinned 128-bit outward-rounded Arb "
            "interval; otherwise returns the first unresolved exact comparison "
            "without claiming an optimum."
        ),
        EuclideanConvexPolygonTriangulationRequest,
        EuclideanConvexPolygonTriangulationResult,
        minimum_euclidean_weight_triangulation,
        "geometry",
        "polygon",
        "triangulation",
        "optimization",
        "euclidean",
        "square-root-sum",
        examples=(
            example(
                "unit_square_euclidean",
                (
                    "Triangulate a unit square under the non-hull Euclidean "
                    "diagonal-length objective; the polygon must be simple and "
                    f"strictly CCW convex with 4 to {MAX_EUCLIDEAN_TRIANGULATION_VERTICES} "
                    "vertices whose complete serialized result, including the "
                    "echoed source ring, stays inside its output bound."
                ),
                {"polygon": {"points": _UNIT_SQUARE}},
            ),
        ),
    ),
    geometry_operation(
        "geometry.polygon.triangulation.minimum_weight.compute",
        "Compute an exact minimum-weight convex-polygon triangulation",
        (
            "Compute the deterministic minimum triangulation of a strict CCW "
            "convex rational polygon under a complete exact rational weight for "
            "each non-hull diagonal, charging every selected diagonal once."
        ),
        ConvexPolygonTriangulationRequest,
        ConvexPolygonTriangulationResult,
        minimum_weight_triangulation,
        "geometry",
        "polygon",
        "triangulation",
        "optimization",
        examples=(
            example(
                "unit_square_complete_diagonals",
                "Triangulate a unit square; use a strict CCW convex polygon and ordered weights for every non-hull diagonal.",
                {
                    "polygon": {"points": _UNIT_SQUARE},
                    "diagonal_weights": [
                        {"first": 0, "second": 2, "weight": {"num": "1", "den": "1"}},
                        {"first": 1, "second": 3, "weight": {"num": "2", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
    geometry_operation(
        "geometry.polygon.simple.decide",
        "Decide exact simple-polygon validity",
        (
            "Decide whether a bounded rational polygon ring is simple and "
            "preserve the first exact violating edge pair when it is not."
        ),
        PolygonRequest,
        SimplePolygonDecisionResult,
        simple_polygon,
        "geometry",
        "polygon",
        "decision",
        examples=(
            example(
                "unit_square_is_simple",
                "Check every edge pair of a unit-square ring; a simple polygon's adjacent edges meet only at endpoints.",
                {"points": _UNIT_SQUARE},
            ),
        ),
    ),
    geometry_operation(
        "geometry.polygon.point.classify",
        "Classify a point against a simple polygon",
        (
            "Classify one rational point as inside, on the boundary of, or "
            "outside one structurally validated simple rational polygon."
        ),
        SimplePolygonPointRequest,
        PolygonPointClassificationResult,
        classify_polygon_point,
        "geometry",
        "polygon",
        "classification",
        examples=(
            example(
                "unit_square_center",
                "Classify the center of a unit square; the polygon must be simple.",
                {
                    "polygon": {"points": _UNIT_SQUARE},
                    "point": {
                        "x": {"num": "1", "den": "2"},
                        "y": {"num": "1", "den": "2"},
                    },
                },
            ),
        ),
    ),
)
