"""Per-producer independent replay declarations for exact geometry."""

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.contracts.geometry import (
    ConvexPolygonTriangulationRequest,
    PointPairRequest,
    PointSetRequest,
    PointTripleRequest,
    PolygonRequest,
    SegmentIntersectionRequest,
    SimplePolygonPointRequest,
)

_ENTRYPOINT = "jacobian_checkers.exact_geometry"
_OPERATIONS = (
    (
        "geometry.polygon.triangulation.minimum_weight.compute",
        ConvexPolygonTriangulationRequest,
    ),
    ("geometry.points.compute.convex_hull", PointSetRequest),
    ("geometry.points.compute.squared_distance", PointPairRequest),
    ("geometry.segment.compute.midpoint", PointPairRequest),
    ("geometry.segments.intersection.compute", SegmentIntersectionRequest),
    ("geometry.polygon.simple.decide", PolygonRequest),
    ("geometry.polygon.point.classify", SimplePolygonPointRequest),
    ("geometry.triangle.compute.orientation", PointTripleRequest),
    ("geometry.triangle.compute.centroid", PointTripleRequest),
)

GEOMETRY_EXACT_REPLAY_CHECKERS = tuple(
    ExactReplayCheckerDeclaration(
        capability_id,
        request_model,
        "check_exact_geometry",
        "geometry.exact_rational_result",
        entrypoint_module=_ENTRYPOINT,
        replay_method="standard-library exact-rational replay",
        reason=(
            "operator-authorized standard-library rational replay independent of "
            "the SymPy geometry producer"
        ),
    )
    for capability_id, request_model in _OPERATIONS
)

__all__ = ["GEOMETRY_EXACT_REPLAY_CHECKERS"]
