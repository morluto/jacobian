"""Public declarations for exact convex-polytope profiles."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.convex._models import (
    ActiveFacetProfileRequest,
    ActiveFacetProfileResult,
    DirectionCoverageRequest,
    DirectionCoverageResult,
    DirectionLocalMotionRequest,
    DirectionLocalMotionResult,
)
from jacobian.math.geometry.convex.operations import (
    active_facet_profile,
    direction_local_motion,
    direction_set_coverage,
)

_SQUARE = {
    "space": {"axes": ["x", "y"]},
    "inequalities": [
        {
            "inequality_id": "bottom",
            "normal": [{"num": "0", "den": "1"}, {"num": "-1", "den": "1"}],
            "bound": {"num": "0", "den": "1"},
        },
        {
            "inequality_id": "left",
            "normal": [{"num": "-1", "den": "1"}, {"num": "0", "den": "1"}],
            "bound": {"num": "0", "den": "1"},
        },
        {
            "inequality_id": "right",
            "normal": [{"num": "1", "den": "1"}, {"num": "0", "den": "1"}],
            "bound": {"num": "1", "den": "1"},
        },
        {
            "inequality_id": "top",
            "normal": [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
            "bound": {"num": "1", "den": "1"},
        },
    ],
}
_POINT = {
    "space": {"axes": ["x", "y"]},
    "coordinates": [{"num": "0", "den": "1"}, {"num": "0", "den": "1"}],
}
_DIRECTION = {
    "space": {"axes": ["x", "y"]},
    "components": [{"num": "1", "den": "1"}, {"num": "1", "den": "1"}],
}
TOOLS = (
    MathTool(
        operation_id="convex_geometry.polytope.active_facet_profile.compute",
        title="Compute the exact active-facet profile of a rational polytope",
        description="Return all exact slacks, active inequalities, and the rank of their outward normals at a boundary point; the point and polytope must share one labelled space.",
        request_type=ActiveFacetProfileRequest,
        result_type=ActiveFacetProfileResult,
        run=lambda r: active_facet_profile(r.polytope, r.point),
        tags=("convex-geometry", "polytope", "facets"),
        discovery_terms=("active facets", "normal cone profile"),
        examples=(
            OperationExample(
                name="square_corner",
                description="Compute the active facets at the square origin; the point must be on the boundary.",
                input={"polytope": _SQUARE, "point": _POINT},
            ),
        ),
    ),
    MathTool(
        operation_id="convex_geometry.polytope.direction_set_coverage.compute",
        title="Compute complete strict direction coverage of boundary points",
        description="Return the complete point-by-direction local-motion matrix, strict coverage, uncovered points, and deterministic witnesses for a finite supplied boundary-point and direction family.",
        request_type=DirectionCoverageRequest,
        result_type=DirectionCoverageResult,
        run=direction_set_coverage,
        tags=("convex-geometry", "illumination", "coverage"),
        discovery_terms=(
            "strict illumination coverage matrix",
            "point direction illumination",
        ),
        examples=(
            OperationExample(
                name="corner_matrix",
                description="Compute the exact local-motion matrix for one square corner and one entering direction; every point and direction must use the square space.",
                input={
                    "polytope": _SQUARE,
                    "points": [{"point_id": "origin", "point": _POINT}],
                    "directions": [
                        {"direction_id": "diagonal", "direction": _DIRECTION}
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="convex_geometry.polytope.direction_local_motion.compute",
        title="Decide exact strict local motion at a polytope boundary point",
        description="For a bounded labelled H-polytope, boundary point, and nonzero rational direction on one space, return exact slacks and active normal pairings classified as strict, tangent, or outward.",
        request_type=DirectionLocalMotionRequest,
        result_type=DirectionLocalMotionResult,
        run=lambda r: direction_local_motion(r.polytope, r.point, r.direction),
        tags=("convex-geometry", "polytope", "illumination", "exact"),
        discovery_terms=(
            "strict illumination direction",
            "polytope boundary local motion",
        ),
        examples=(
            OperationExample(
                name="square_corner_strict_entry",
                description="Decide strict entry of the square origin along (1,1); the point must be on the boundary and the direction nonzero.",
                input={"polytope": _SQUARE, "point": _POINT, "direction": _DIRECTION},
            ),
        ),
    ),
)
__all__ = ["TOOLS"]
