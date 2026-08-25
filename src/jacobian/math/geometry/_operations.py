"""Exact rational planar-geometry operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry._models import (
    CircleInversionRequest,
    CircumcircleRequest,
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    CircumradiusTripleEntry,
    ClosedSegment2D,
    CollinearTripleWitness,
    ConcyclicQuadrupleWitness,
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
    PolygonIntersectionWitness,
    PolygonPointClassificationResult,
    PolygonRequest,
    RationalPoint2D,
    SegmentIntersectionRequest,
    SegmentIntersectionResult,
    SimplePolygonDecisionResult,
    SimplePolygonPointRequest,
)
from jacobian.math.geometry._predicates import are_collinear, determinant4

Compute = Callable[[LinePairRequest], GeometryBooleanResult]


def _fraction(value: Any) -> Fraction:
    import sympy

    rational = sympy.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


def _wire_rational(value: Any) -> CanonicalRational:
    fraction = _fraction(value)
    return CanonicalRational(
        num=format_canonical_integer(fraction.numerator),
        den=format_canonical_integer(fraction.denominator),
    )


def _point(value: RationalPoint2D) -> Any:
    import sympy
    from sympy.geometry import Point2D

    return Point2D(
        sympy.Rational(value.x.as_fraction()),
        sympy.Rational(value.y.as_fraction()),
    )


def _wire_point(value: Any) -> RationalPoint2D:
    return RationalPoint2D(
        x=_wire_rational(value.x),
        y=_wire_rational(value.y),
    )


def _canonical_points(values: tuple[Any, ...]) -> tuple[RationalPoint2D, ...]:
    if len(values) <= 2:
        values = tuple(sorted(values, key=lambda point: (point.x, point.y)))
    else:
        doubled_area = sum(
            left.x * right.y - left.y * right.x
            for left, right in zip(values, values[1:] + values[:1], strict=True)
        )
        if doubled_area < 0:
            values = tuple(reversed(values))
        start = min(
            range(len(values)), key=lambda index: (values[index].x, values[index].y)
        )
        values = values[start:] + values[:start]
    return tuple(_wire_point(point) for point in values)


def _closed_segment(value: ClosedSegment2D) -> tuple[Any, Any]:
    return _point(value.start), _point(value.end)


def _cross(left: tuple[Any, Any], right: tuple[Any, Any]) -> Any:
    return left[0] * right[1] - left[1] * right[0]


def _subtract(left: Any, right: Any) -> tuple[Any, Any]:
    return left.x - right.x, left.y - right.y


def _on_segment(point: Any, start: Any, end: Any) -> bool:
    return bool(
        _cross(_subtract(point, start), _subtract(end, start)) == 0
        and min(start.x, end.x) <= point.x <= max(start.x, end.x)
        and min(start.y, end.y) <= point.y <= max(start.y, end.y)
    )


def _pair_points(request: PointPairRequest) -> tuple[Any, Any]:
    pair = request
    return _point(pair.first), _point(pair.second)


def _line(value: LineRequest) -> Any:
    from sympy.geometry import Line2D

    return Line2D(_point(value.first), _point(value.second))


def squared_distance(request: PointPairRequest) -> GeometryRationalResult:
    first, second = _pair_points(request)
    return GeometryRationalResult(value=_wire_rational(first.distance(second) ** 2))


def midpoint(request: PointPairRequest) -> GeometryPointResult:
    first, second = _pair_points(request)
    return GeometryPointResult(point=_wire_point(first.midpoint(second)))


def segment_intersection(
    request: SegmentIntersectionRequest,
) -> SegmentIntersectionResult:
    import sympy
    from sympy.geometry import Point2D

    pair = request
    first_start, first_end = _closed_segment(pair.first)
    second_start, second_end = _closed_segment(pair.second)
    first_degenerate = first_start == first_end
    second_degenerate = second_start == second_end
    if first_degenerate or second_degenerate:
        if first_degenerate and _on_segment(first_start, second_start, second_end):
            point = first_start
        elif second_degenerate and _on_segment(second_start, first_start, first_end):
            point = second_start
        else:
            return SegmentIntersectionResult(status="DISJOINT")
        return SegmentIntersectionResult(
            status="POINT",
            point=_wire_point(point),
            contact_kind="DEGENERATE_TOUCH",
        )

    first_direction = _subtract(first_end, first_start)
    second_direction = _subtract(second_end, second_start)
    denominator = _cross(first_direction, second_direction)
    offset = _subtract(second_start, first_start)
    if denominator != 0:
        first_parameter = sympy.cancel(_cross(offset, second_direction) / denominator)
        second_parameter = sympy.cancel(_cross(offset, first_direction) / denominator)
        if not (0 <= first_parameter <= 1 and 0 <= second_parameter <= 1):
            return SegmentIntersectionResult(status="DISJOINT")
        point = Point2D(
            first_start.x + first_parameter * first_direction[0],
            first_start.y + first_parameter * first_direction[1],
        )
        return SegmentIntersectionResult(
            status="POINT",
            point=_wire_point(point),
            contact_kind=(
                "PROPER"
                if 0 < first_parameter < 1 and 0 < second_parameter < 1
                else "ENDPOINT_TOUCH"
            ),
        )
    if _cross(offset, first_direction) != 0:
        return SegmentIntersectionResult(status="DISJOINT")
    common = tuple(
        sorted(
            {
                point
                for point in (
                    first_start,
                    first_end,
                    second_start,
                    second_end,
                )
                if _on_segment(point, first_start, first_end)
                and _on_segment(point, second_start, second_end)
            },
            key=lambda point: (point.x, point.y),
        )
    )
    if not common:
        return SegmentIntersectionResult(status="DISJOINT")
    if len(common) == 1:
        return SegmentIntersectionResult(
            status="POINT",
            point=_wire_point(common[0]),
            contact_kind="ENDPOINT_TOUCH",
        )
    return SegmentIntersectionResult(
        status="OVERLAP",
        overlap=ClosedSegment2D(
            start=_wire_point(common[0]),
            end=_wire_point(common[-1]),
        ),
    )


def collinear(request: PointTripleRequest) -> GeometryBooleanResult:
    from sympy.geometry import Point2D

    triple = request
    return GeometryBooleanResult(
        holds=Point2D.is_collinear(
            _point(triple.first),
            _point(triple.second),
            _point(triple.third),
        )
    )


def concyclic(request: PointQuadrupleRequest) -> GeometryBooleanResult:
    from sympy.geometry import Point2D

    points = request
    return GeometryBooleanResult(
        holds=Point2D.is_concyclic(
            _point(points.first),
            _point(points.second),
            _point(points.third),
            _point(points.fourth),
        )
    )


def line_predicate(
    predicate: Callable[[Any, Any], bool],
) -> Compute:
    def compute(request: LinePairRequest) -> GeometryBooleanResult:
        pair = request
        return GeometryBooleanResult(
            holds=predicate(_line(pair.first_line), _line(pair.second_line))
        )

    return compute


def line_intersection(request: LinePairRequest) -> GeometryLineIntersectionResult:
    from sympy.geometry import Point2D

    pair = request
    first, second = _line(pair.first_line), _line(pair.second_line)
    if first.equals(second):
        return GeometryLineIntersectionResult(status="COINCIDENT")
    intersections = first.intersection(second)
    if not intersections:
        return GeometryLineIntersectionResult(status="PARALLEL")
    point = intersections[0]
    if not isinstance(point, Point2D):
        raise ValueError("line intersection did not produce one exact point")
    return GeometryLineIntersectionResult(status="POINT", point=_wire_point(point))


def projection(request: PointLineRequest) -> GeometryPointResult:
    from sympy.geometry import Point2D

    value = request
    projected = _line(value.line).projection(_point(value.point))
    if not isinstance(projected, Point2D):
        raise ValueError("line projection did not produce one exact point")
    return GeometryPointResult(point=_wire_point(projected))


def orientation(request: PointTripleRequest) -> GeometryOrientationResult:
    import sympy

    triple = request
    first, second, third = (
        _point(triple.first),
        _point(triple.second),
        _point(triple.third),
    )
    determinant = (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (
        third.x - first.x
    )
    return GeometryOrientationResult(
        orientation=cast(Any, int(sympy.sign(determinant)))
    )


def centroid(request: PointTripleRequest) -> GeometryPointResult:
    from sympy.geometry import Point2D

    triple = request
    points = [_point(triple.first), _point(triple.second), _point(triple.third)]
    return GeometryPointResult(
        point=_wire_point(
            Point2D(
                sum(point.x for point in points) / 3,
                sum(point.y for point in points) / 3,
            )
        )
    )


def circumcircle(request: CircumcircleRequest) -> GeometryCircleResult:
    from sympy.geometry import Circle

    triple = request
    points = [_point(triple.first), _point(triple.second), _point(triple.third)]
    circle = Circle(*points)
    return GeometryCircleResult(
        center=_wire_point(circle.center),
        radius_squared=_wire_rational(circle.radius**2),
    )


def signed_area(request: PolygonRequest) -> GeometryRationalResult:
    from fractions import Fraction

    polygon = request
    points = [_point(point) for point in polygon.points]
    # Shoelace formula: works for any polygon including degenerate/collinear ones.
    total = Fraction(0)
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += Fraction(current.x * following.y - current.y * following.x)
    return GeometryRationalResult(value=_wire_rational(total / 2))


def simple_polygon(request: PolygonRequest) -> SimplePolygonDecisionResult:
    polygon = request
    points = polygon.points
    checked = 0
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            checked += 1
            intersection = segment_intersection(
                SegmentIntersectionRequest(
                    first=ClosedSegment2D(
                        start=points[first],
                        end=points[(first + 1) % len(points)],
                    ),
                    second=ClosedSegment2D(
                        start=points[second],
                        end=points[(second + 1) % len(points)],
                    ),
                )
            )
            adjacent = (first - second) % len(points) in {1, len(points) - 1}
            shared = (
                points[0] if (first, second) == (0, len(points) - 1) else points[second]
            )
            valid = (
                intersection.status == "POINT"
                and intersection.point == shared
                and intersection.contact_kind == "ENDPOINT_TOUCH"
                if adjacent
                else intersection.status == "DISJOINT"
            )
            if not valid:
                return SimplePolygonDecisionResult(
                    vertex_count=len(points),
                    is_simple=False,
                    checked_edge_pairs=checked,
                    witness=PolygonIntersectionWitness(
                        first_edge_index=first,
                        second_edge_index=second,
                        intersection=intersection,
                    ),
                )
    return SimplePolygonDecisionResult(
        vertex_count=len(points),
        is_simple=True,
        checked_edge_pairs=checked,
    )


def classify_polygon_point(
    request: SimplePolygonPointRequest,
) -> PolygonPointClassificationResult:
    from sympy.geometry import Polygon

    value = request
    point = _point(value.point)
    points = tuple(_point(item) for item in value.polygon.points)
    for index, start in enumerate(points):
        if _on_segment(point, start, points[(index + 1) % len(points)]):
            return PolygonPointClassificationResult(
                polygon_vertex_count=len(points),
                classification="BOUNDARY",
                boundary_edge_index=index,
            )
    polygon = Polygon(*points)
    return PolygonPointClassificationResult(
        polygon_vertex_count=len(points),
        classification=("INSIDE" if polygon.encloses_point(point) else "OUTSIDE"),
    )


def convex_hull_points(request: PointSetRequest) -> GeometryConvexHullResult:
    from sympy.geometry import Line2D, Point2D, Polygon, Segment2D
    from sympy.geometry.util import convex_hull

    point_set = request
    hull = convex_hull(*(_point(point) for point in point_set.points))
    if isinstance(hull, Point2D):
        points = (hull,)
    elif isinstance(hull, (Line2D, Segment2D)):
        points = tuple(
            sorted(
                cast(tuple[Point2D, Point2D], hull.points),
                key=lambda point: (point.x, point.y),
            )
        )
    else:
        points = tuple(cast(Polygon, hull).vertices)
    return GeometryConvexHullResult(points=_canonical_points(points))


def circle_inversion(request: CircleInversionRequest) -> GeometryPointResult:
    """Invert one rational planar point in a circle.

    Returns ``I_{c,s}(p) = c + (s / ||p - c||^2) * (p - c)`` exactly over the
    rationals, where ``c`` is the center and ``s`` is the positive squared
    inversion radius.  The request ``p == c`` is rejected before division.
    """
    center = request.center
    point = request.point
    power = request.power.as_fraction()
    dx = point.x.as_fraction() - center.x.as_fraction()
    dy = point.y.as_fraction() - center.y.as_fraction()
    norm_squared = dx * dx + dy * dy
    if norm_squared == 0:
        raise ValueError("the point to invert must differ from the center")
    scale = power / norm_squared
    inverted = RationalPoint2D(
        x=CanonicalRational.from_fraction(center.x.as_fraction() + scale * dx),
        y=CanonicalRational.from_fraction(center.y.as_fraction() + scale * dy),
    )
    return GeometryPointResult(point=inverted)


# ---------------------------------------------------------------------------
# Configuration-level operations (issues #2107, #2106)
# ---------------------------------------------------------------------------


def _points_to_fractions(
    points: tuple[RationalPoint2D, ...],
) -> list[tuple[Fraction, Fraction]]:
    return [(p.x.as_fraction(), p.y.as_fraction()) for p in points]


def general_position_search(request: GeneralPositionRequest) -> GeneralPositionResult:
    """Find all collinear triples and concyclic quadruples in a point configuration."""
    from itertools import combinations

    pts = _points_to_fractions(request.points)
    n = len(pts)

    collinear_triples: list[CollinearTripleWitness] = []
    collinear_set: set[tuple[int, int, int]] = set()
    for i, j, k in combinations(range(n), 3):
        if are_collinear(pts[i], pts[j], pts[k]):
            collinear_triples.append(CollinearTripleWitness(indices=(i, j, k)))
            collinear_set.add((i, j, k))

    concyclic_quadruples: list[ConcyclicQuadrupleWitness] = []
    for i, j, k, m in combinations(range(n), 4):
        # Exclude collinear quadruples: any 3 collinear points are degenerate
        # (collinear points lie on a line, not a finite circle). The
        # determinant criterion |x^2+y^2, x, y, 1| is zero for four collinear
        # points, but no Euclidean circle contains three distinct collinear
        # points, so such quadruples must not be reported as concyclic.
        if (
            (i, j, k) in collinear_set
            or (i, j, m) in collinear_set
            or (i, k, m) in collinear_set
            or (j, k, m) in collinear_set
        ):
            continue
        a, b, c, d = pts[i], pts[j], pts[k], pts[m]
        rows: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
        for px, py in (a, b, c, d):
            rows.append((px * px + py * py, px, py, Fraction(1)))
        determinant = determinant4(tuple(rows))
        if determinant == 0:
            concyclic_quadruples.append(ConcyclicQuadrupleWitness(indices=(i, j, k, m)))

    return GeneralPositionResult(
        points=request.points,
        num_points=n,
        has_collinear_triple=bool(collinear_triples),
        has_concyclic_quadruple=bool(concyclic_quadruples),
        collinear_triples=tuple(collinear_triples),
        concyclic_quadruples=tuple(concyclic_quadruples),
    )


def circumradius_profile(
    request: CircumradiusProfileRequest,
) -> CircumradiusProfileResult:
    """Compute circumradius squared for every unordered triple in a configuration."""
    from fractions import Fraction
    from itertools import combinations

    pts = _points_to_fractions(request.points)
    n = len(pts)

    entries: list[CircumradiusTripleEntry] = []
    for i, j, k in combinations(range(n), 3):
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        if cross == 0:
            entries.append(
                CircumradiusTripleEntry(
                    indices=(i, j, k),
                    is_degenerate=True,
                )
            )
        else:
            ab_sq = (bx - ax) ** 2 + (by - ay) ** 2
            bc_sq = (cx - bx) ** 2 + (cy - by) ** 2
            ca_sq = (ax - cx) ** 2 + (ay - cy) ** 2
            r_sq = Fraction(ab_sq * bc_sq * ca_sq) / Fraction(4 * cross * cross)
            entries.append(
                CircumradiusTripleEntry(
                    indices=(i, j, k),
                    is_degenerate=False,
                    radius_squared=_wire_rational(r_sq),
                )
            )

    # Ensure deterministic lexicographic order (combinations already yields sorted).
    entries.sort(key=lambda e: e.indices)
    return CircumradiusProfileResult(
        points=request.points,
        num_points=n,
        entries=tuple(entries),
    )
