"""Exact rational planar-geometry operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math._rational_height import RationalHeight
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
    LabelledPoint2D,
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


def _minor3(
    m: list[list[Fraction]],
    r0: int,
    r1: int,
    r2: int,
    c0: int,
    c1: int,
    c2: int,
) -> Fraction:
    """Determinant of the 3x3 submatrix on rows r* and columns c*."""
    return (
        m[r0][c0] * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1])
        - m[r0][c1] * (m[r1][c0] * m[r2][c2] - m[r1][c2] * m[r2][c0])
        + m[r0][c2] * (m[r1][c0] * m[r2][c1] - m[r1][c1] * m[r2][c0])
    )


def _concyclic_determinant(
    xy: list[tuple[Fraction, Fraction]], quad: tuple[int, int, int, int]
) -> Fraction:
    """Determinant of [x^2+y^2, x, y, 1] over one quadruple of points.

    Vanishing is equivalent to concyclicity once a non-collinear triple
    exists; the row-0 Laplace expansion keeps every cofactor on rows 1-3.
    """
    m = [
        [
            xy[index][0] * xy[index][0] + xy[index][1] * xy[index][1],
            xy[index][0],
            xy[index][1],
            Fraction(1),
        ]
        for index in quad
    ]
    return (
        m[0][0] * _minor3(m, 1, 2, 3, 1, 2, 3)
        - m[0][1] * _minor3(m, 1, 2, 3, 0, 2, 3)
        + m[0][2] * _minor3(m, 1, 2, 3, 0, 1, 3)
        - m[0][3] * _minor3(m, 1, 2, 3, 0, 1, 2)
    )


def _screen_configuration(
    pts: tuple[LabelledPoint2D, ...],
) -> tuple[
    bool,
    bool,
    tuple[int, int, int] | None,
    tuple[int, int, int, int] | None,
    int,
    int,
]:
    """Complete bounded forbidden-pattern enumeration over distinct points.

    Returns ``(has_collinear, has_concyclic, collinear_indices,
    concyclic_indices, checked_triples, checked_quadruples)`` so execution and
    result validation replay identical bounded work. A vanishing concyclic
    determinant also covers collinear quadruples; such degenerate quadruples
    lie on no circle and are skipped rather than reported.
    """
    from itertools import combinations

    n = len(pts)
    xy = [(item.point.x.as_fraction(), item.point.y.as_fraction()) for item in pts]

    def _collinear(a: int, b: int, c: int) -> bool:
        xa, ya = xy[a]
        xb, yb = xy[b]
        xc, yc = xy[c]
        return (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa) == 0

    collinear_indices = None
    checked_triples = 0
    for i, j, k in combinations(range(n), 3):
        checked_triples += 1
        if _collinear(i, j, k):
            collinear_indices = (i, j, k)
            break

    concyclic_indices = None
    checked_quadruples = 0
    for i, j, k, ell in combinations(range(n), 4):
        checked_quadruples += 1
        if _concyclic_determinant(xy, (i, j, k, ell)) != 0:
            continue
        if (
            _collinear(i, j, k)
            and _collinear(i, j, ell)
            and _collinear(i, k, ell)
            and _collinear(j, k, ell)
        ):
            continue
        concyclic_indices = (i, j, k, ell)
        break

    return (
        collinear_indices is not None,
        concyclic_indices is not None,
        collinear_indices,
        concyclic_indices,
        checked_triples,
        checked_quadruples,
    )


def forbidden_patterns(request: ForbiddenPatternsRequest) -> ForbiddenPatternsResult:
    """Find a collinear triple or a nondegenerate concyclic quadruple.

    Three configuration points are collinear when the exact determinant
    ``(x_b-x_a)(y_c-y_a) - (y_b-y_a)(x_c-x_a)`` vanishes; four points are
    concyclic when the exact ``[x^2+y^2, x, y, 1]`` determinant vanishes and
    some triple among them is non-collinear. Either witness ends the bounded
    enumeration; otherwise every triple and quadruple is checked exactly.
    """
    from jacobian.math.geometry._models import (
        CollinearTriple,
        ConcyclicQuadruple,
    )

    pts = request.configuration.points
    (
        has_collinear,
        has_concyclic,
        collinear_indices,
        concyclic_indices,
        checked_triples,
        checked_quadruples,
    ) = _screen_configuration(pts)

    return ForbiddenPatternsResult(
        configuration=request.configuration,
        point_count=len(pts),
        has_collinear_triple=has_collinear,
        has_concyclic_quadruple=has_concyclic,
        collinear_triple=(
            CollinearTriple(
                first=collinear_indices[0],
                second=collinear_indices[1],
                third=collinear_indices[2],
            )
            if collinear_indices is not None
            else None
        ),
        concyclic_quadruple=(
            ConcyclicQuadruple(
                first=concyclic_indices[0],
                second=concyclic_indices[1],
                third=concyclic_indices[2],
                fourth=concyclic_indices[3],
            )
            if concyclic_indices is not None
            else None
        ),
        checked_triples=checked_triples,
        checked_quadruples=checked_quadruples,
    )


def circumradius_profile(
    request: CircumradiusProfileRequest,
) -> CircumradiusProfileResult:
    """Compute exact circumradius data for every unordered triple.

    A nondegenerate triangle carries its exact squared circumradius
    ``R^2 = (|a-b|^2 |b-c|^2 |c-a|^2) / (4 * cross(a,b,c)^2)``; a collinear
    triple is flagged degenerate with no circumcircle. The admitted input
    height bounds every derived entry inside CanonicalRational's digit limit,
    and each computed value is re-checked before it enters the result.
    """
    from itertools import combinations

    points = request.points
    coords = [
        (item.point.x.as_fraction(), item.point.y.as_fraction()) for item in points
    ]
    entries: list[CircumradiusTripleEntry] = []
    for i, j, k in combinations(range(len(points)), 3):
        ax, ay = coords[i]
        bx, by = coords[j]
        cx, cy = coords[k]
        dab = (ax - bx) ** 2 + (ay - by) ** 2
        dbc = (bx - cx) ** 2 + (by - cy) ** 2
        dac = (ax - cx) ** 2 + (ay - cy) ** 2
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        labels = (points[i].label, points[j].label, points[k].label)
        if cross == 0:
            entries.append(
                CircumradiusTripleEntry(
                    labels=labels,
                    indices=(i, j, k),
                    collinear=True,
                )
            )
            continue
        squared = (dab * dbc * dac) / (4 * cross * cross)
        canonical = CanonicalRational.from_fraction(squared)
        if RationalHeight.from_canonical(canonical).exceeds(
            MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise ValueError(
                "squared circumradius exceeds the canonical rational limit"
            )
        entries.append(
            CircumradiusTripleEntry(
                labels=labels,
                indices=(i, j, k),
                collinear=False,
                squared_circumradius=canonical,
            )
        )
    return CircumradiusProfileResult(
        point_count=len(points),
        triple_count=len(entries),
        entries=tuple(entries),
        points=tuple(points),
    )
