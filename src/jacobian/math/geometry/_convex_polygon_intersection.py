"""Exact intersection of two strict convex rational polygons."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry._models import (
    ClosedSegment2D,
    GeometryConvexHullResult,
    RationalPoint2D,
)

MAX_CONVEX_POLYGON_VERTICES = 64
MAX_CONVEX_INTERSECTION_COORDINATE_DIGITS = 256
MAX_CONVEX_INTERSECTION_VERTICES = 128

CONVEX_KIND = Literal["EMPTY", "POINT", "SEGMENT", "POLYGON"]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"geometry.{reason}", message)


def _point_to_fraction(p: RationalPoint2D) -> tuple[Fraction, Fraction]:
    return p.x.as_fraction(), p.y.as_fraction()


def _fraction_to_point(x: Fraction, y: Fraction) -> RationalPoint2D:
    from jacobian.canonical import format_canonical_integer

    return RationalPoint2D(
        x=CanonicalRational(
            num=format_canonical_integer(x.numerator),
            den=format_canonical_integer(x.denominator),
        ),
        y=CanonicalRational(
            num=format_canonical_integer(y.numerator),
            den=format_canonical_integer(y.denominator),
        ),
    )


def _cross(ax: Fraction, ay: Fraction, bx: Fraction, by: Fraction) -> Fraction:
    return ax * by - ay * bx


def _cross_points(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    c: tuple[Fraction, Fraction],
) -> Fraction:
    # (b-a) x (c-b)
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    return abx * bcy - aby * bcx


def _cross_edge(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    p: tuple[Fraction, Fraction],
) -> Fraction:
    # (b-a) x (p-a)
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    return abx * apy - aby * apx


def _on_segment_frac(
    p: tuple[Fraction, Fraction],
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
) -> bool:
    if _cross_edge(a, b, p) != 0:
        return False
    # check within bounding box inclusive
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[
        1
    ] <= max(a[1], b[1])


def _line_intersection(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
    c: tuple[Fraction, Fraction],
    d: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction] | None:
    # Line AB and CD intersection (not segments)
    # Solve a + t*(b-a) = c + u*(d-c)
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    cdx = d[0] - c[0]
    cdy = d[1] - c[1]
    cross_ab_cd = abx * cdy - aby * cdx
    if cross_ab_cd == 0:
        return None  # parallel
    acx = c[0] - a[0]
    acy = c[1] - a[1]
    t = (acx * cdy - acy * cdx) / cross_ab_cd
    # For exact, compute fraction
    # But using Fraction division already exact
    x = a[0] + t * abx
    y = a[1] + t * aby
    return (x, y)


def _is_inside_half_plane(
    p: tuple[Fraction, Fraction],
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
) -> bool:
    # For CCW polygon, interior is left of edge a->b (cross >=0)
    return _cross_edge(a, b, p) >= 0


def _clip_polygon_by_edge(
    subject: list[tuple[Fraction, Fraction]],
    clip_a: tuple[Fraction, Fraction],
    clip_b: tuple[Fraction, Fraction],
) -> list[tuple[Fraction, Fraction]]:
    if not subject:
        return []
    output: list[tuple[Fraction, Fraction]] = []
    prev = subject[-1]
    prev_inside = _is_inside_half_plane(prev, clip_a, clip_b)
    for curr in subject:
        curr_inside = _is_inside_half_plane(curr, clip_a, clip_b)
        if curr_inside:
            if not prev_inside:
                inter = _line_intersection(prev, curr, clip_a, clip_b)
                if inter is not None:
                    output.append(inter)
            output.append(curr)
        elif prev_inside:
            inter = _line_intersection(prev, curr, clip_a, clip_b)
            if inter is not None:
                output.append(inter)
        prev = curr
        prev_inside = curr_inside
    return output


def _deduplicate_points(
    points: list[tuple[Fraction, Fraction]],
) -> list[tuple[Fraction, Fraction]]:
    if not points:
        return []
    dedup: list[tuple[Fraction, Fraction]] = [points[0]]
    for p in points[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    # Check wrap-around duplicate (first and last same)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup


def _canonical_polygon_rotation(
    points: list[tuple[Fraction, Fraction]],
) -> list[tuple[Fraction, Fraction]]:
    # Find least vertex lexicographically (x then y)
    min_idx = min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    return points[min_idx:] + points[:min_idx]


class ConvexRationalPolygon(StrictModel):
    """Strict convex polygon with rational vertices in CCW order."""

    vertices: tuple[RationalPoint2D, ...] = Field(
        min_length=3,
        max_length=MAX_CONVEX_POLYGON_VERTICES,
        description=(
            "Strict CCW vertices with no three consecutive collinear. "
            f"At least 3 and at most {MAX_CONVEX_POLYGON_VERTICES} vertices."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw(cls, value: object) -> object:
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_strict_convex_ccw(self) -> Self:
        n = len(self.vertices)
        if n < 3:
            raise _validation_error(
                "convex_polygon_min_vertices",
                "convex polygon must have at least 3 vertices",
            )
        # Check distinct
        keys = tuple((v.x.num, v.x.den, v.y.num, v.y.den) for v in self.vertices)
        if len(set(keys)) != n:
            raise _validation_error(
                "convex_polygon_vertices_not_unique",
                "convex polygon vertices must be distinct",
            )
        # Check bounded digits
        for v in self.vertices:
            require_bounded_rational(
                v.x,
                max_digits=MAX_CONVEX_INTERSECTION_COORDINATE_DIGITS,
                label="vertex x",
            )
            require_bounded_rational(
                v.y,
                max_digits=MAX_CONVEX_INTERSECTION_COORDINATE_DIGITS,
                label="vertex y",
            )
        # Check strict CCW and convex (all left turns)
        pts = [_point_to_fraction(v) for v in self.vertices]
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            c = pts[(i + 2) % n]
            cross = _cross_points(a, b, c)
            if cross <= 0:
                raise _validation_error(
                    "convex_polygon_not_strict_ccw",
                    "convex polygon must be strictly CCW with no three consecutive collinear "
                    f"(cross at index {i + 1} is {cross})",
                )
        # Local left turns alone admit some star-shaped rings.  A convex CCW
        # ring has every vertex in every directed edge's closed left half-plane.
        for edge_index in range(n):
            a = pts[edge_index]
            b = pts[(edge_index + 1) % n]
            if any(_cross_edge(a, b, point) < 0 for point in pts):
                raise _validation_error(
                    "convex_polygon_not_simple",
                    "convex polygon vertices must lie in every directed edge's left half-plane",
                )
        return self

    @classmethod
    def from_convex_hull(cls, hull: GeometryConvexHullResult) -> Self:
        """Explicitly map the shared hull value into this strict input domain."""
        return cls(vertices=hull.points)


class ConvexPolygonIntersectionRequest(StrictModel):
    polygon_a: ConvexRationalPolygon
    polygon_b: ConvexRationalPolygon

    @model_validator(mode="before")
    @classmethod
    def bound_raw(cls, value: object) -> object:
        return canonicalize_json_containers(value)


class ConvexPolygonIntersectionResult(StrictModel):
    """Discriminated intersection of two convex polygons with provenance."""

    polygon_a: ConvexRationalPolygon
    polygon_b: ConvexRationalPolygon
    kind: CONVEX_KIND
    point: RationalPoint2D | None = None
    segment: ClosedSegment2D | None = None
    polygon: GeometryConvexHullResult | None = None
    # For each output vertex, the active edge pairs (poly_idx 0/1, edge_idx)
    # For EMPTY, empty; for POINT/SEGMENT/POLYGON, matches vertex count.
    vertex_active_edges: tuple[tuple[tuple[int, int], ...], ...] = Field(
        default=(),
        description="For each output vertex, the tight source edges as (poly_idx, edge_idx).",
    )

    @model_validator(mode="after")
    def require_discriminated(self) -> Self:  # noqa: C901
        if self.kind == "EMPTY":
            if (
                self.point is not None
                or self.segment is not None
                or self.polygon is not None
            ):
                raise _validation_error(
                    "convex_intersection_empty_payload",
                    "EMPTY must have no point/segment/polygon",
                )
            if self.vertex_active_edges != ():
                raise _validation_error(
                    "convex_intersection_empty_active",
                    "EMPTY must have no active edges",
                )
        elif self.kind == "POINT":
            if (
                self.point is None
                or self.segment is not None
                or self.polygon is not None
            ):
                raise _validation_error(
                    "convex_intersection_point_payload",
                    "POINT must have exactly point",
                )
            if len(self.vertex_active_edges) != 1:
                raise _validation_error(
                    "convex_intersection_point_active",
                    "POINT must have one vertex active edge tuple",
                )
        elif self.kind == "SEGMENT":
            if (
                self.segment is None
                or self.point is not None
                or self.polygon is not None
            ):
                raise _validation_error(
                    "convex_intersection_segment_payload",
                    "SEGMENT must have exactly segment",
                )
            if len(self.vertex_active_edges) != 2:
                raise _validation_error(
                    "convex_intersection_segment_active",
                    "SEGMENT must have two vertex active edge tuples",
                )
            # Check segment endpoints lexicographically ordered
            a = _point_to_fraction(self.segment.start)
            b = _point_to_fraction(self.segment.end)
            if a >= b:
                raise _validation_error(
                    "convex_intersection_segment_order",
                    "segment endpoints must be lexicographically ordered",
                )
        elif self.kind == "POLYGON":
            if (
                self.polygon is None
                or self.point is not None
                or self.segment is not None
            ):
                raise _validation_error(
                    "convex_intersection_polygon_payload",
                    "POLYGON must have exactly polygon",
                )
            if len(self.vertex_active_edges) != len(self.polygon.points):
                raise _validation_error(
                    "convex_intersection_polygon_active",
                    "POLYGON active edges must match polygon vertex count",
                )
            if len(self.polygon.points) < 3:
                raise _validation_error(
                    "convex_intersection_polygon_dimension",
                    "POLYGON must contain at least three vertices",
                )
            # Check polygon is strict CCW and canonical least vertex
            pts = [_point_to_fraction(p) for p in self.polygon.points]
            # Check least vertex
            min_idx = min(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
            if min_idx != 0:
                raise _validation_error(
                    "convex_intersection_polygon_canonical",
                    "polygon must start with lexicographically least vertex",
                )
        else:
            raise _validation_error("convex_intersection_kind", "unknown kind")
        rings = (self.polygon_a.vertices, self.polygon_b.vertices)
        for row_index, active_edges in enumerate(self.vertex_active_edges):
            if not active_edges or tuple(sorted(set(active_edges))) != active_edges:
                raise _validation_error(
                    "convex_intersection_active_edges",
                    f"active-edge row {row_index} must be nonempty, distinct, and sorted",
                )
            for polygon_index, edge_index in active_edges:
                if polygon_index not in (0, 1) or not 0 <= edge_index < len(
                    rings[polygon_index]
                ):
                    raise _validation_error(
                        "convex_intersection_active_edge_range",
                        f"active edge ({polygon_index}, {edge_index}) does not belong to a source ring",
                    )
        return self

    @classmethod
    def _from_kernel(
        cls,
        polygon_a: ConvexRationalPolygon,
        polygon_b: ConvexRationalPolygon,
        kind: CONVEX_KIND,
        point: RationalPoint2D | None = None,
        segment: ClosedSegment2D | None = None,
        polygon: GeometryConvexHullResult | None = None,
        vertex_active_edges: tuple[tuple[tuple[int, int], ...], ...] = (),
    ) -> Self:
        return cls.model_construct(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind=kind,
            point=point,
            segment=segment,
            polygon=polygon,
            vertex_active_edges=vertex_active_edges,
        )


def _admit_convex_polygon_intersection(
    polygon_a: ConvexRationalPolygon, polygon_b: ConvexRationalPolygon
) -> None:
    n = len(polygon_a.vertices)
    m = len(polygon_b.vertices)
    if n > MAX_CONVEX_POLYGON_VERTICES or m > MAX_CONVEX_POLYGON_VERTICES:
        raise OperationDomainValidationError(
            location=("polygon_a", "vertices"),
            code="geometry.convex_polygon.vertex_count",
            message=f"convex polygon vertex count exceeds {MAX_CONVEX_POLYGON_VERTICES}",
        )
    # Check coordinate growth for intersection: worst-case output vertices <= n+m
    if n + m > MAX_CONVEX_INTERSECTION_VERTICES:
        raise OperationDomainValidationError(
            location=("polygon_a",),
            code="geometry.convex_polygon.output_vertex_count",
            message=f"predicted output vertices {n + m} exceeds {MAX_CONVEX_INTERSECTION_VERTICES}",
        )


def _active_edges_for_vertices(
    output_fracs: list[tuple[Fraction, Fraction]],
    poly_a_fracs: list[tuple[Fraction, Fraction]],
    poly_b_fracs: list[tuple[Fraction, Fraction]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    # For each output point, find tight edges from both polygons
    result: list[tuple[tuple[int, int], ...]] = []
    n = len(poly_a_fracs)
    m = len(poly_b_fracs)
    # Build edge lists
    edges_a = [(poly_a_fracs[i], poly_a_fracs[(i + 1) % n]) for i in range(n)]
    edges_b = [(poly_b_fracs[i], poly_b_fracs[(i + 1) % m]) for i in range(m)]
    for p in output_fracs:
        active: list[tuple[int, int]] = []
        for ei, (a, b) in enumerate(edges_a):
            if _on_segment_frac(p, a, b):
                active.append((0, ei))
        for ei, (a, b) in enumerate(edges_b):
            if _on_segment_frac(p, a, b):
                active.append((1, ei))
        # Also consider if point equals a vertex, it will be on two edges incident; that's fine
        # Ensure deterministic order: sort by (poly_idx, edge_idx)
        active_sorted = tuple(sorted(active))
        # Deduplicate
        result.append(active_sorted)
    return tuple(result)


def convex_polygon_intersection(  # noqa: C901
    polygon_a: ConvexRationalPolygon, polygon_b: ConvexRationalPolygon
) -> ConvexPolygonIntersectionResult:
    _admit_convex_polygon_intersection(polygon_a, polygon_b)
    frac_a = [_point_to_fraction(v) for v in polygon_a.vertices]
    frac_b = [_point_to_fraction(v) for v in polygon_b.vertices]

    # Sutherland-Hodgman: clip A by B
    output = frac_a
    for i in range(len(frac_b)):
        clip_a = frac_b[i]
        clip_b = frac_b[(i + 1) % len(frac_b)]
        output = _clip_polygon_by_edge(output, clip_a, clip_b)
        if not output:
            break

    output = _deduplicate_points(output)
    # Handle EMPTY
    if not output:
        return ConvexPolygonIntersectionResult._from_kernel(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind="EMPTY",
            vertex_active_edges=(),
        )
    if len(output) == 1:
        pt = _fraction_to_point(output[0][0], output[0][1])
        active = _active_edges_for_vertices(output, frac_a, frac_b)
        return ConvexPolygonIntersectionResult._from_kernel(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind="POINT",
            point=pt,
            vertex_active_edges=active,
        )
    if len(output) == 2:
        # Check if they are distinct
        if output[0] == output[1]:
            pt = _fraction_to_point(output[0][0], output[0][1])
            active = _active_edges_for_vertices([output[0]], frac_a, frac_b)
            return ConvexPolygonIntersectionResult._from_kernel(
                polygon_a=polygon_a,
                polygon_b=polygon_b,
                kind="POINT",
                point=pt,
                vertex_active_edges=active,
            )
        # Sort lexicographically
        p0, p1 = output
        if p0 > p1:
            p0, p1 = p1, p0
            output = [p0, p1]
        seg = ClosedSegment2D(
            start=_fraction_to_point(p0[0], p0[1]),
            end=_fraction_to_point(p1[0], p1[1]),
        )
        active = _active_edges_for_vertices(output, frac_a, frac_b)
        return ConvexPolygonIntersectionResult._from_kernel(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind="SEGMENT",
            segment=seg,
            vertex_active_edges=active,
        )
    # For polygon, check if points are collinear? With strict convex inputs, output with >=3 points should be polygon, but may have collinear consecutive points due to clipping. Remove collinear.
    # Remove collinear consecutive triples
    # First, ensure polygon is not degenerate due to all points collinear (should have been segment)
    # Check if all points collinear
    # For >=3, we need to ensure strict convex output: remove collinear middle points
    filtered = output
    # Iteratively remove collinear middle points
    changed = True
    while changed and len(filtered) >= 3:
        changed = False
        new_filtered: list[tuple[Fraction, Fraction]] = []
        n = len(filtered)
        for i in range(n):
            prev = filtered[(i - 1) % n]
            curr = filtered[i]
            nxt = filtered[(i + 1) % n]
            cross = _cross_points(prev, curr, nxt)
            if cross == 0:
                # collinear, remove curr
                changed = True
                continue
            new_filtered.append(curr)
        if changed:
            filtered = new_filtered
            if len(filtered) < 3:
                break
    if len(filtered) == 0:
        return ConvexPolygonIntersectionResult._from_kernel(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind="EMPTY",
            vertex_active_edges=(),
        )
    if len(filtered) == 1:
        pt = _fraction_to_point(filtered[0][0], filtered[0][1])
        active = _active_edges_for_vertices(filtered, frac_a, frac_b)
        return ConvexPolygonIntersectionResult._from_kernel(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind="POINT",
            point=pt,
            vertex_active_edges=active,
        )
    if len(filtered) == 2:
        p0, p1 = filtered
        if p0 > p1:
            p0, p1 = p1, p0
            filtered = [p0, p1]
        seg = ClosedSegment2D(
            start=_fraction_to_point(filtered[0][0], filtered[0][1]),
            end=_fraction_to_point(filtered[1][0], filtered[1][1]),
        )
        active = _active_edges_for_vertices(filtered, frac_a, frac_b)
        return ConvexPolygonIntersectionResult._from_kernel(
            polygon_a=polygon_a,
            polygon_b=polygon_b,
            kind="SEGMENT",
            segment=seg,
            vertex_active_edges=active,
        )
    # Now we have polygon with >=3 points, should be strict convex CCW
    # Ensure CCW orientation: clipping preserves CCW, but after deduplication and filtering, should remain CCW.
    # Canonicalize rotation to least vertex
    filtered = _canonical_polygon_rotation(filtered)
    # Convert to RationalPoint2D list
    points = tuple(_fraction_to_point(x, y) for x, y in filtered)
    # Build GeometryConvexHullResult for canonical polygon (reuse its validation)
    # GeometryConvexHullResult expects points in CCW and canonical, but we have filtered already CCW and canonical
    poly_result = GeometryConvexHullResult(points=points)
    active = _active_edges_for_vertices(filtered, frac_a, frac_b)
    return ConvexPolygonIntersectionResult._from_kernel(
        polygon_a=polygon_a,
        polygon_b=polygon_b,
        kind="POLYGON",
        polygon=poly_result,
        vertex_active_edges=active,
    )
