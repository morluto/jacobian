"""Exact rational planar-geometry operations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry._models import (
    INVERSION_ADMISSION_DIGITS,
    MAX_COORDINATE_DIGITS,
    CircumradiusProfileResult,
    CircumradiusTripleEntry,
    ClosedSegment2D,
    CollinearTripleWitness,
    ConcyclicQuadrupleWitness,
    GeneralPositionResult,
    GeometryBooleanResult,
    GeometryCircleResult,
    GeometryConvexHullResult,
    GeometryLineIntersectionResult,
    GeometryOrientationResult,
    GeometryPointResult,
    GeometryRationalResult,
    PolygonIntersectionWitness,
    PolygonPointClassificationResult,
    RationalLine2D,
    RationalPoint2D,
    SegmentIntersectionResult,
    SimplePolygonDecisionResult,
    _inverted_components_within_bound,
    _is_simple_ring,
    _point_key,
    _require_bounded_configuration,
    _require_circumradius_output_bound,
    _require_general_position_work_bound,
    _require_inversion_admission_bound,
)
from jacobian.math.geometry._predicates import are_collinear, determinant4

__all__ = [
    "centroid",
    "circle_inversion",
    "circumcircle",
    "circumradius_profile",
    "classify_polygon_point",
    "collinear",
    "concyclic",
    "convex_hull_points",
    "general_position_search",
    "line_intersection",
    "line_predicate",
    "midpoint",
    "orientation",
    "projection",
    "segment_intersection",
    "signed_area",
    "simple_polygon",
    "squared_distance",
]


def _reject_geometry_domain(
    *, location: tuple[str | int, ...], code: str, message: str
) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _admit_inversion(
    center: RationalPoint2D, power: CanonicalRational, point: RationalPoint2D
) -> None:
    if power.as_fraction() <= 0:
        _reject_geometry_domain(
            location=("power",),
            code="geometry.inversion_power_must_be_positive",
            message="inversion power must be a positive rational",
        )
    if point == center:
        _reject_geometry_domain(
            location=("point",),
            code="geometry.inversion_point_must_differ_from_center",
            message="the point to invert must differ from the center",
        )

    for value, location, label in (
        (center.x, ("center", "x"), "inversion center x"),
        (center.y, ("center", "y"), "inversion center y"),
        (power, ("power",), "inversion power"),
        (point.x, ("point", "x"), "point x"),
        (point.y, ("point", "y"), "point y"),
    ):
        try:
            _require_inversion_admission_bound(value, label)
        except ValueError as exc:
            _reject_geometry_domain(
                location=location,
                code="geometry.inversion_input_digit_bound",
                message=str(exc),
            )

    if not _inverted_components_within_bound(
        center,
        power,
        point,
        INVERSION_ADMISSION_DIGITS,
    ):
        _reject_geometry_domain(
            location=("point",),
            code="geometry.inversion_result_digit_bound",
            message=(
                "circle inversion result exceeds the "
                f"{INVERSION_ADMISSION_DIGITS}-digit circle-inversion admission bound"
            ),
        )


def _admit_configuration(
    points: tuple[RationalPoint2D, ...], *, output_bound: bool
) -> None:
    try:
        _require_bounded_configuration(points)
    except ValueError as exc:
        for index, point in enumerate(points):
            for axis, value in (("x", point.x), ("y", point.y)):
                if (
                    max(len(value.num.lstrip("-")), len(value.den))
                    > MAX_COORDINATE_DIGITS
                ):
                    _reject_geometry_domain(
                        location=("points", index, axis),
                        code="geometry.coordinate_digits_max",
                        message=str(exc),
                    )
        _reject_geometry_domain(
            location=("points",),
            code="geometry.configuration_coordinate_bound",
            message=str(exc),
        )

    try:
        if output_bound:
            _require_circumradius_output_bound(points)
        else:
            _require_general_position_work_bound(points)
    except ValueError as exc:
        code = (
            "geometry.circumradius_profile_n_points_max_digits"
            if output_bound
            else "geometry.general_position_search_n_points_max"
        )
        _reject_geometry_domain(location=("points",), code=code, message=str(exc))


def _admit_circumcircle(
    points: tuple[RationalPoint2D, RationalPoint2D, RationalPoint2D],
) -> None:
    keys = tuple(_point_key(point) for point in points)
    if len(set(keys)) != len(keys):
        _reject_geometry_domain(
            location=("first", "second", "third"),
            code="geometry.circumcircle_requires_three_distinct_points",
            message="circumcircle requires three distinct points",
        )
    if are_collinear(*keys):
        _reject_geometry_domain(
            location=("first", "second", "third"),
            code="geometry.circumcircle_requires_three_noncollinear_points",
            message="circumcircle requires three noncollinear points",
        )


def _admit_simple_polygon_point(
    points: tuple[RationalPoint2D, ...],
) -> None:
    if not _is_simple_ring(points):
        _reject_geometry_domain(
            location=("polygon", "points"),
            code="geometry.point_classification_requires_a_simple_polygon",
            message="point classification requires a simple polygon",
        )


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


def _line(value: RationalLine2D) -> Any:
    from sympy.geometry import Line2D

    return Line2D(_point(value.first), _point(value.second))


def squared_distance(
    first_point: RationalPoint2D, second_point: RationalPoint2D
) -> GeometryRationalResult:
    first, second = _point(first_point), _point(second_point)
    return GeometryRationalResult(value=_wire_rational(first.distance(second) ** 2))


def midpoint(
    first_point: RationalPoint2D, second_point: RationalPoint2D
) -> GeometryPointResult:
    first, second = _point(first_point), _point(second_point)
    return GeometryPointResult(point=_wire_point(first.midpoint(second)))


def segment_intersection(
    first_segment: ClosedSegment2D, second_segment: ClosedSegment2D
) -> SegmentIntersectionResult:
    import sympy
    from sympy.geometry import Point2D

    first_start, first_end = _closed_segment(first_segment)
    second_start, second_end = _closed_segment(second_segment)
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


def collinear(
    first_point: RationalPoint2D,
    second_point: RationalPoint2D,
    third_point: RationalPoint2D,
) -> GeometryBooleanResult:
    from sympy.geometry import Point2D

    return GeometryBooleanResult(
        holds=Point2D.is_collinear(
            _point(first_point), _point(second_point), _point(third_point)
        )
    )


def concyclic(
    first_point: RationalPoint2D,
    second_point: RationalPoint2D,
    third_point: RationalPoint2D,
    fourth_point: RationalPoint2D,
) -> GeometryBooleanResult:
    from sympy.geometry import Point2D

    return GeometryBooleanResult(
        holds=Point2D.is_concyclic(
            _point(first_point),
            _point(second_point),
            _point(third_point),
            _point(fourth_point),
        )
    )


def line_predicate(
    predicate: Callable[[Any, Any], bool],
) -> Callable[[RationalLine2D, RationalLine2D], GeometryBooleanResult]:
    def compute(
        first_line: RationalLine2D, second_line: RationalLine2D
    ) -> GeometryBooleanResult:
        return GeometryBooleanResult(
            holds=predicate(_line(first_line), _line(second_line))
        )

    return compute


def line_intersection(
    first_line: RationalLine2D, second_line: RationalLine2D
) -> GeometryLineIntersectionResult:
    from sympy.geometry import Point2D

    first, second = _line(first_line), _line(second_line)
    if first.equals(second):
        return GeometryLineIntersectionResult(status="COINCIDENT")
    intersections = first.intersection(second)
    if not intersections:
        return GeometryLineIntersectionResult(status="PARALLEL")
    point = intersections[0]
    if not isinstance(point, Point2D):
        raise ValueError("line intersection did not produce one exact point")
    return GeometryLineIntersectionResult(status="POINT", point=_wire_point(point))


def projection(
    point_value: RationalPoint2D, line_value: RationalLine2D
) -> GeometryPointResult:
    from sympy.geometry import Point2D

    projected = _line(line_value).projection(_point(point_value))
    if not isinstance(projected, Point2D):
        raise ValueError("line projection did not produce one exact point")
    return GeometryPointResult(point=_wire_point(projected))


def orientation(
    first_point: RationalPoint2D,
    second_point: RationalPoint2D,
    third_point: RationalPoint2D,
) -> GeometryOrientationResult:
    import sympy

    first, second, third = (
        _point(first_point),
        _point(second_point),
        _point(third_point),
    )
    determinant = (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (
        third.x - first.x
    )
    return GeometryOrientationResult(
        orientation=cast(Any, int(sympy.sign(determinant)))
    )


def centroid(
    first_point: RationalPoint2D,
    second_point: RationalPoint2D,
    third_point: RationalPoint2D,
) -> GeometryPointResult:
    from sympy.geometry import Point2D

    points = [_point(first_point), _point(second_point), _point(third_point)]
    return GeometryPointResult(
        point=_wire_point(
            Point2D(
                sum(point.x for point in points) / 3,
                sum(point.y for point in points) / 3,
            )
        )
    )


def circumcircle(
    first_point: RationalPoint2D,
    second_point: RationalPoint2D,
    third_point: RationalPoint2D,
) -> GeometryCircleResult:
    from sympy.geometry import Circle

    _admit_circumcircle((first_point, second_point, third_point))
    points = [_point(first_point), _point(second_point), _point(third_point)]
    circle = Circle(*points)
    # SymPy can represent a circle with very large rational coordinates as a
    # Segment2D instead of raising an error.  The three points have already
    # passed exact non-collinearity admission, so retain the mathematical
    # postcondition with the owner-local rational formula in that case.
    if not hasattr(circle, "center") or not hasattr(circle, "radius"):
        first, second, third = (
            _point_key(point) for point in (first_point, second_point, third_point)
        )
        cross = (
            first[0] * (second[1] - third[1])
            + second[0] * (third[1] - first[1])
            + third[0] * (first[1] - second[1])
        )
        first_norm = first[0] * first[0] + first[1] * first[1]
        second_norm = second[0] * second[0] + second[1] * second[1]
        third_norm = third[0] * third[0] + third[1] * third[1]
        center = (
            (
                first_norm * (second[1] - third[1])
                + second_norm * (third[1] - first[1])
                + third_norm * (first[1] - second[1])
            )
            / (2 * cross),
            (
                first_norm * (third[0] - second[0])
                + second_norm * (first[0] - third[0])
                + third_norm * (second[0] - first[0])
            )
            / (2 * cross),
        )
        radius_squared = (center[0] - first[0]) ** 2 + (center[1] - first[1]) ** 2
        return GeometryCircleResult(
            center=RationalPoint2D(
                x=_wire_rational(center[0]), y=_wire_rational(center[1])
            ),
            radius_squared=_wire_rational(radius_squared),
        )
    return GeometryCircleResult(
        center=_wire_point(circle.center),
        radius_squared=_wire_rational(circle.radius**2),
    )


def signed_area(points: tuple[RationalPoint2D, ...]) -> GeometryRationalResult:
    fraction_points = _points_to_fractions(points)
    # Shoelace formula: works for any polygon including degenerate/collinear ones.
    total = Fraction(0)
    for index, current in enumerate(fraction_points):
        following = fraction_points[(index + 1) % len(fraction_points)]
        total += current[0] * following[1] - current[1] * following[0]
    return GeometryRationalResult(value=_wire_rational(total / 2))


def simple_polygon(points: tuple[RationalPoint2D, ...]) -> SimplePolygonDecisionResult:
    checked = 0
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            checked += 1
            intersection = segment_intersection(
                ClosedSegment2D(
                    start=points[first],
                    end=points[(first + 1) % len(points)],
                ),
                ClosedSegment2D(
                    start=points[second],
                    end=points[(second + 1) % len(points)],
                ),
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
    point_value: RationalPoint2D,
    polygon_points: tuple[RationalPoint2D, ...],
) -> PolygonPointClassificationResult:
    from sympy.geometry import Polygon

    _admit_simple_polygon_point(polygon_points)
    point = _point(point_value)
    points = tuple(_point(item) for item in polygon_points)
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


def convex_hull_points(
    point_values: tuple[RationalPoint2D, ...],
) -> GeometryConvexHullResult:
    from sympy.geometry import Line2D, Point2D, Polygon, Segment2D
    from sympy.geometry.util import convex_hull

    hull = convex_hull(*(_point(point) for point in point_values))
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


def circle_inversion(
    center: RationalPoint2D,
    power: CanonicalRational,
    point: RationalPoint2D,
) -> GeometryPointResult:
    """Invert one rational planar point in a circle.

    Returns ``I_{c,s}(p) = c + (s / ||p - c||^2) * (p - c)`` exactly over the
    rationals, where ``c`` is the center and ``s`` is the positive squared
    inversion radius.  The request ``p == c`` is rejected before division.
    """
    _admit_inversion(center, power, point)
    power_value = power.as_fraction()
    dx = point.x.as_fraction() - center.x.as_fraction()
    dy = point.y.as_fraction() - center.y.as_fraction()
    norm_squared = dx * dx + dy * dy
    scale = power_value / norm_squared
    inverted = RationalPoint2D(
        x=CanonicalRational.from_fraction(center.x.as_fraction() + scale * dx),
        y=CanonicalRational.from_fraction(center.y.as_fraction() + scale * dy),
    )
    return GeometryPointResult(point=inverted)


# ---------------------------------------------------------------------------
# Configuration-level operations
# ---------------------------------------------------------------------------


def _points_to_fractions(
    points: tuple[RationalPoint2D, ...],
) -> list[tuple[Fraction, Fraction]]:
    return [(p.x.as_fraction(), p.y.as_fraction()) for p in points]


def general_position_search(
    points: tuple[RationalPoint2D, ...],
) -> GeneralPositionResult:
    """Find all collinear triples and concyclic quadruples in a point configuration."""
    from itertools import combinations

    _admit_configuration(points, output_bound=False)
    pts = _points_to_fractions(points)
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

    return GeneralPositionResult._from_kernel(
        points=points,
        collinear_triples=tuple(collinear_triples),
        concyclic_quadruples=tuple(concyclic_quadruples),
    )


def circumradius_profile(
    points: tuple[RationalPoint2D, ...],
) -> CircumradiusProfileResult:
    """Compute circumradius squared for every unordered triple in a configuration."""
    from fractions import Fraction
    from itertools import combinations

    _admit_configuration(points, output_bound=True)
    pts = _points_to_fractions(points)
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
    return CircumradiusProfileResult._from_kernel(
        points=points,
        entries=tuple(entries),
    )
