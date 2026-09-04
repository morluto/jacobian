"""Polygon-owned exact geometry operations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry._convex_polygon_intersection import (
    ConvexPolygonIntersectionRequest,
    ConvexPolygonIntersectionResult,
)
from jacobian.math.geometry._convex_polygon_intersection import (
    convex_polygon_intersection as _convex_polygon_intersection_kernel,
)
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
from jacobian.math.geometry._tools import (
    classify_polygon_point,
    simple_polygon,
)
from jacobian.math.geometry._triangulation import minimum_weight_triangulation


def _run_convex_polygon_intersection(
    request: ConvexPolygonIntersectionRequest,
) -> ConvexPolygonIntersectionResult:
    return _convex_polygon_intersection_kernel(request.polygon_a, request.polygon_b)


_UNIT_SQUARE = [
    {"x": {"num": "0", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "1", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "1", "den": "1"}, "y": {"num": "1", "den": "1"}},
    {"x": {"num": "0", "den": "1"}, "y": {"num": "1", "den": "1"}},
]

POLYGON_OPERATIONS: MathTools = (
    MathTool(
        operation_id="geometry.polygon.triangulation.minimum_euclidean_weight.compute",
        title="Compute a certified minimum Euclidean convex-polygon triangulation",
        description=(
            "Compute one deterministic minimum triangulation of a strict CCW convex "
            f"simple rational polygon of 4 to {MAX_EUCLIDEAN_TRIANGULATION_VERTICES} "
            "vertices, charging each selected "
            "non-hull diagonal once by its Euclidean length; admission bounds "
            "the dynamic-program work, exact expression terms, and retained "
            "diagonal cardinality. Returns "
            "the exact "
            "sum-of-square-roots cost expression only when every finite DP "
            "comparison is separated by a pinned 128-bit outward-rounded Arb "
            "interval; otherwise returns the first unresolved exact comparison "
            "without claiming an optimum."
        ),
        request_type=EuclideanConvexPolygonTriangulationRequest,
        result_type=EuclideanConvexPolygonTriangulationResult,
        run=minimum_euclidean_weight_triangulation,
        tags=(
            "geometry",
            "polygon",
            "triangulation",
            "optimization",
            "euclidean",
            "square-root-sum",
        ),
        examples=(
            OperationExample(
                name="unit_square_euclidean",
                description=(
                    "Triangulate a unit square under the non-hull Euclidean "
                    "diagonal-length objective; the polygon must be simple and "
                    f"strictly CCW convex with 4 to {MAX_EUCLIDEAN_TRIANGULATION_VERTICES} "
                    "vertices."
                ),
                input={"polygon": {"points": _UNIT_SQUARE}},
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.polygon.triangulation.minimum_weight.compute",
        title="Compute an exact minimum-weight convex-polygon triangulation",
        description=(
            "Compute the deterministic minimum triangulation of a strict CCW "
            "convex rational polygon under a complete exact rational weight for "
            "each non-hull diagonal, charging every selected diagonal once."
        ),
        request_type=ConvexPolygonTriangulationRequest,
        result_type=ConvexPolygonTriangulationResult,
        run=minimum_weight_triangulation,
        tags=("geometry", "polygon", "triangulation", "optimization"),
        examples=(
            OperationExample(
                name="unit_square_complete_diagonals",
                description="Triangulate a unit square; use a strict CCW convex polygon and ordered weights for every non-hull diagonal.",
                input={
                    "polygon": {"points": _UNIT_SQUARE},
                    "diagonal_weights": [
                        {"first": 0, "second": 2, "weight": {"num": "1", "den": "1"}},
                        {"first": 1, "second": 3, "weight": {"num": "2", "den": "1"}},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.polygon.simple.decide",
        title="Decide exact simple-polygon validity",
        description=(
            "Decide whether a bounded rational polygon ring is simple and "
            "preserve the first exact violating edge pair when it is not."
        ),
        request_type=PolygonRequest,
        result_type=SimplePolygonDecisionResult,
        run=simple_polygon,
        tags=("geometry", "polygon", "decision"),
        examples=(
            OperationExample(
                name="unit_square_is_simple",
                description="Check every edge pair of a unit-square ring; a simple polygon's adjacent edges meet only at endpoints.",
                input={"points": _UNIT_SQUARE},
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.polygon.point.classify",
        title="Classify a point against a simple polygon",
        description=(
            "Classify one rational point as inside, on the boundary of, or "
            "outside one structurally validated simple rational polygon."
        ),
        request_type=SimplePolygonPointRequest,
        result_type=PolygonPointClassificationResult,
        run=classify_polygon_point,
        tags=("geometry", "polygon", "classification"),
        examples=(
            OperationExample(
                name="unit_square_center",
                description="Classify the center of a unit square; the polygon must be simple.",
                input={
                    "polygon": {"points": _UNIT_SQUARE},
                    "point": {
                        "x": {"num": "1", "den": "2"},
                        "y": {"num": "1", "den": "2"},
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="geometry.convex_polygon.intersection.compute",
        title="Compute the exact intersection of two convex polygons",
        description=(
            "For two strict CCW convex rational polygons with at least three "
            "vertices, return their exact closed-set intersection as EMPTY, a "
            "single rational point, a maximal lexicographically ordered segment, "
            "or a strict CCW polygon beginning with its least vertex. The "
            f"operation admits at most {64} vertices per polygon, output at most "
            "128 vertices, and uses exact Sutherland-Hodgman clipping with "
            "Fractions and provenance of tight source edges."
        ),
        request_type=ConvexPolygonIntersectionRequest,
        result_type=ConvexPolygonIntersectionResult,
        run=_run_convex_polygon_intersection,
        tags=("geometry", "convex-polygon", "intersection", "exact"),
        examples=(
            OperationExample(
                name="overlapping_squares",
                description=(
                    "Intersect [0,2]^2 with [1,3]x[1,3]; both polygons must be "
                    "strict CCW convex and the result is the polygon [1,2]^2."
                ),
                input={
                    "polygon_a": {
                        "vertices": [
                            {
                                "x": {"num": "0", "den": "1"},
                                "y": {"num": "0", "den": "1"},
                            },
                            {
                                "x": {"num": "2", "den": "1"},
                                "y": {"num": "0", "den": "1"},
                            },
                            {
                                "x": {"num": "2", "den": "1"},
                                "y": {"num": "2", "den": "1"},
                            },
                            {
                                "x": {"num": "0", "den": "1"},
                                "y": {"num": "2", "den": "1"},
                            },
                        ]
                    },
                    "polygon_b": {
                        "vertices": [
                            {
                                "x": {"num": "1", "den": "1"},
                                "y": {"num": "1", "den": "1"},
                            },
                            {
                                "x": {"num": "3", "den": "1"},
                                "y": {"num": "1", "den": "1"},
                            },
                            {
                                "x": {"num": "3", "den": "1"},
                                "y": {"num": "3", "den": "1"},
                            },
                            {
                                "x": {"num": "1", "den": "1"},
                                "y": {"num": "3", "den": "1"},
                            },
                        ]
                    },
                },
            ),
        ),
    ),
)
