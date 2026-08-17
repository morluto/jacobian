"""Polygon-owned exact geometry operations."""

from jacobian.catalog._examples import example
from jacobian.math.geometry._models import (
    ConvexPolygonTriangulationRequest,
    ConvexPolygonTriangulationResult,
    GeometryRationalResult,
    PolygonPointClassificationResult,
    PolygonRequest,
    SimplePolygonDecisionResult,
    SimplePolygonPointRequest,
)
from jacobian.math.geometry._operations import (
    classify_polygon_point,
    signed_area,
    simple_polygon,
)
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry._triangulation import minimum_weight_triangulation

_UNIT_SQUARE = [
    {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
    {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
]

POLYGON_OPERATIONS = (
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
        "geometry.polygon.compute.signed_area",
        "Compute polygon signed area",
        "Compute exact oriented area of a simple rational polygon.",
        PolygonRequest,
        GeometryRationalResult,
        signed_area,
        "geometry",
        "polygon",
        examples=(
            example(
                "unit_square_signed_area",
                "Compute the signed area of a unit square.",
                {"points": _UNIT_SQUARE},
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
