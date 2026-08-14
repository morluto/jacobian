"""Per-producer independent replay declarations for exact geometry."""

from jacobian.checker_operations import AuthorizedChecker
from jacobian.contracts.geometry import (
    ConvexPolygonTriangulationRequest,
    PointPairRequest,
    PointSetRequest,
    PointTripleRequest,
    PolygonRequest,
    SegmentIntersectionRequest,
    SimplePolygonPointRequest,
)
from jacobian.contracts.operations import (
    ProviderInstallTier,
    ProviderObservation,
)
from jacobian.provider_runtime import source_provider_runtime

_ENTRYPOINT = "jacobian_checkers.exact_geometry"


def _geometry_runtime(*, checker_ids: tuple[str, ...] = ()) -> ProviderObservation:
    return source_provider_runtime(
        "jacobian.exact-geometry-checker",
        version="1",
        entrypoint="jacobian_checkers.exact_geometry:check_exact_geometry",
        install_tier=ProviderInstallTier.T1,
        license_id="MIT",
        features=("standard-library-rational-replay", "clean-process-checker"),
        checker_ids=checker_ids,
    )


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

GEOMETRY_AUTHORIZED_CHECKERS = tuple(
    AuthorizedChecker(
        operation_id,
        request_model,
        "check_exact_geometry",
        "geometry.exact_rational_result",
        entrypoint_module=_ENTRYPOINT,
        observation_loader=_geometry_runtime,
        replay_method="standard-library exact-rational replay",
        reason=(
            "operator-authorized standard-library rational replay independent of "
            "the SymPy geometry producer"
        ),
    )
    for operation_id, request_model in _OPERATIONS
)

__all__ = ["GEOMETRY_AUTHORIZED_CHECKERS"]
