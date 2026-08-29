"""Exact rational planar-geometry operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.geometry import operations as _native
from jacobian.math.geometry._models import (
    CircleInversionRequest,
    CircumcircleRequest,
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    GeneralPositionRequest,
    GeneralPositionResult,
    GeometryBooleanResult,
    GeometryCircleResult,
    GeometryConvexHullResult,
    GeometryLineIntersectionResult,
    GeometryOrientationResult,
    GeometryPointResult,
    GeometryRationalResult,
    LinePairRequest,
    LineRequest,
    PointLineRequest,
    PointPairRequest,
    PointQuadrupleRequest,
    PointSetRequest,
    PointTripleRequest,
    PolygonPointClassificationResult,
    PolygonRequest,
    RationalLine2D,
    SegmentIntersectionRequest,
    SegmentIntersectionResult,
    SimplePolygonDecisionResult,
    SimplePolygonPointRequest,
)


def _line_value(request: LineRequest) -> RationalLine2D:
    """Project the validated wire line onto the canonical line value."""
    return RationalLine2D.model_construct(first=request.first, second=request.second)


def squared_distance(request: PointPairRequest) -> GeometryRationalResult:
    return _native.squared_distance(request.first, request.second)


def midpoint(request: PointPairRequest) -> GeometryPointResult:
    return _native.midpoint(request.first, request.second)


def segment_intersection(
    request: SegmentIntersectionRequest,
) -> SegmentIntersectionResult:
    return _native.segment_intersection(request.first, request.second)


def collinear(request: PointTripleRequest) -> GeometryBooleanResult:
    return _native.collinear(request.first, request.second, request.third)


def concyclic(request: PointQuadrupleRequest) -> GeometryBooleanResult:
    return _native.concyclic(
        request.first, request.second, request.third, request.fourth
    )


def line_intersection(request: LinePairRequest) -> GeometryLineIntersectionResult:
    return _native.line_intersection(
        _line_value(request.first_line), _line_value(request.second_line)
    )


def projection(request: PointLineRequest) -> GeometryPointResult:
    return _native.projection(request.point, _line_value(request.line))


def orientation(request: PointTripleRequest) -> GeometryOrientationResult:
    return _native.orientation(request.first, request.second, request.third)


def centroid(request: PointTripleRequest) -> GeometryPointResult:
    return _native.centroid(request.first, request.second, request.third)


def circumcircle(request: CircumcircleRequest) -> GeometryCircleResult:
    return _native.circumcircle(request.first, request.second, request.third)


def signed_area(request: PolygonRequest) -> GeometryRationalResult:
    return _native.signed_area(request.points)


def simple_polygon(request: PolygonRequest) -> SimplePolygonDecisionResult:
    return _native.simple_polygon(request.points)


def classify_polygon_point(
    request: SimplePolygonPointRequest,
) -> PolygonPointClassificationResult:
    return _native.classify_polygon_point(request.point, request.polygon.points)


def convex_hull_points(request: PointSetRequest) -> GeometryConvexHullResult:
    return _native.convex_hull_points(request.points)


def circle_inversion(request: CircleInversionRequest) -> GeometryPointResult:
    return _native.circle_inversion(request.center, request.power, request.point)


def general_position_search(request: GeneralPositionRequest) -> GeneralPositionResult:
    return _native.general_position_search(request.points)


def circumradius_profile(
    request: CircumradiusProfileRequest,
) -> CircumradiusProfileResult:
    return _native.circumradius_profile(request.points)


from jacobian.math.geometry._configuration import CONFIGURATION_OPERATIONS  # noqa: E402
from jacobian.math.geometry._lines import LINE_OPERATIONS  # noqa: E402
from jacobian.math.geometry._points import POINT_OPERATIONS  # noqa: E402
from jacobian.math.geometry._polygons import POLYGON_OPERATIONS  # noqa: E402
from jacobian.math.geometry._segments import SEGMENT_OPERATIONS  # noqa: E402
from jacobian.math.geometry._triangles import TRIANGLE_OPERATIONS  # noqa: E402

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *POINT_OPERATIONS,
    *SEGMENT_OPERATIONS,
    *LINE_OPERATIONS,
    *TRIANGLE_OPERATIONS,
    *POLYGON_OPERATIONS,
    *CONFIGURATION_OPERATIONS,
)
