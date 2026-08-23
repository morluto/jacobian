"""Exact rational planar-geometry operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry._models import (
    CircumcircleRequest,
    CircumradiusProfileRequest,
    CircumradiusProfileResult,
    CircumradiusTripleEntry,
    ClosedSegment2D,
    ForbiddenPatternsRequest,
    ForbiddenPatternsResult,
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


def circumradius_profile(
    request: CircumradiusProfileRequest,
) -> CircumradiusProfileResult:
    """Compute exact circumradius data for every unordered triple of a planar configuration.

    Each triple is either nondegenerate (exact squared circumradius) or
    degenerate (collinear, no circumcircle).
    """
    from itertools import combinations

    points = request.points
    n = len(points)
    coords: list[tuple[Fraction, Fraction]] = [
        (item.point.x.as_fraction(), item.point.y.as_fraction()) for item in points
    ]
    entries: list[CircumradiusTripleEntry] = []
    for i, j, k in combinations(range(n), 3):
        (ax, ay), (bx, by), (cx, cy) = coords[i], coords[j], coords[k]
        # Squared side lengths of the triangle.
        dab = (ax - bx) ** 2 + (ay - by) ** 2
        dbc = (bx - cx) ** 2 + (by - cy) ** 2
        dac = (ax - cx) ** 2 + (ay - cy) ** 2
        # Twice the signed area (cross product) of the triangle.
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross == 0:
            entries.append(
                CircumradiusTripleEntry(
                    labels=(points[i].label, points[j].label, points[k].label),
                    indices=(i, j, k),
                    collinear=True,
                )
            )
            continue
        # R^2 = (|a-b|^2 |b-c|^2 |c-a|^2) / (4 * (2*area)^2)
        squared_circumradius = (dab * dbc * dac) / (4 * cross * cross)
        entries.append(
            CircumradiusTripleEntry(
                labels=(points[i].label, points[j].label, points[k].label),
                indices=(i, j, k),
                collinear=False,
                squared_circumradius=CanonicalRational.from_fraction(
                    squared_circumradius
                ),
            )
        )
    return CircumradiusProfileResult(
        points=request.points,
        point_count=n,
        triple_count=len(entries),
        entries=tuple(entries),
    )


def forbidden_patterns(
    request: ForbiddenPatternsRequest,
) -> ForbiddenPatternsResult:
    """Find a collinear triple or concyclic quadruple, or establish neither exists.

    Three points are collinear when the 2x2 cross-product determinant

        (x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)

    vanishes.  Four points are concyclic when the 4x4 determinant of the rows
    [x^2 + y^2, x, y, 1] vanishes.  Both predicates are evaluated with exact
    ``fractions.Fraction`` arithmetic!
    """
    from itertools import combinations

    from jacobian.math.geometry._models import (
        CollinearTriple,
        ConcyclicQuadruple,
        ForbiddenPatternsResult,
    )

    pts = request.configuration.points
    n = len(pts)
    xy = [(entry.point.x.as_fraction(), entry.point.y.as_fraction()) for entry in pts]

    collinear_triple = None
    has_collinear = False
    checked_triples = 0
    for i, j, k in combinations(range(n), 3):
        checked_triples += 1
        xi, yi = xy[i]
        xj, yj = xy[j]
        xk, yk = xy[k]
        determinant = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        if determinant == 0:
            has_collinear = True
            collinear_triple = CollinearTriple(first=i, second=j, third=k)
            break

    concyclic_quadruple = None
    has_concyclic = False
    checked_quadruples = 0
    for i, j, k, ell in combinations(range(n), 4):
        checked_quadruples += 1
        xi, yi = xy[i]
        xj, yj = xy[j]
        xk, yk = xy[k]
        xl, yl = xy[ell]
        si = xi * xi + yi * yi
        sj = xj * xj + yj * yj
        sk = xk * xk + yk * yk
        sl = xl * xl + yl * yl
        # 4x4 determinant of [x^2+y^2, x, y, 1]
        m = [
            [si, xi, yi, 1],
            [sj, xj, yj, 1],
            [sk, xk, yk, 1],
            [sl, xl, yl, 1],
        ]
        # Laplace expansion of the 4x4 determinant along row 0:
        # each cofactor deletes row 0 and its own column, keeping rows 1-3.
        det = (
            m[0][0] * _minor3(m, 1, 2, 3, 1, 2, 3)
            - m[0][1] * _minor3(m, 1, 2, 3, 0, 2, 3)
            + m[0][2] * _minor3(m, 1, 2, 3, 0, 1, 3)
            - m[0][3] * _minor3(m, 1, 2, 3, 0, 1, 2)
        )
        if det == 0:
            # Collinear quadruples also give det==0 but are not concyclic;
            # require the 4 points not be collinear (at least one triple noncollinear).
            is_collinear_quad = True
            for a, b, c in ((i, j, k), (i, j, ell), (i, k, ell), (j, k, ell)):
                xa, ya = xy[a]
                xb, yb = xy[b]
                xc, yc = xy[c]
                if (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa) != 0:
                    is_collinear_quad = False
                    break
            if is_collinear_quad:
                continue
            has_concyclic = True
            concyclic_quadruple = ConcyclicQuadruple(
                first=i, second=j, third=k, fourth=ell
            )
            break

    return ForbiddenPatternsResult(
        configuration=request.configuration,
        point_count=n,
        has_collinear_triple=has_collinear,
        has_concyclic_quadruple=has_concyclic,
        collinear_triple=collinear_triple,
        concyclic_quadruple=concyclic_quadruple,
        checked_triples=checked_triples,
        checked_quadruples=checked_quadruples,
    )


def _minor3(
    m: list[list[Any]],
    r0: int,
    r1: int,
    r2: int,
    c0: int,
    c1: int,
    c2: int,
) -> Any:
    """3x3 determinant of selected rows and columns."""
    return (
        m[r0][c0] * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1])
        - m[r0][c1] * (m[r1][c0] * m[r2][c2] - m[r1][c2] * m[r2][c0])
        + m[r0][c2] * (m[r1][c0] * m[r2][c1] - m[r1][c1] * m[r2][c0])
    )
