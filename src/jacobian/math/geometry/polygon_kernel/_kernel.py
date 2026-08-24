"""Bounded exact half-plane kernel for simple counterclockwise polygons."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.math.geometry._models import (
    GeometryConvexHullResult,
    PointSetRequest,
    RationalPoint2D,
)
from jacobian.math.geometry._operations import convex_hull_points
from jacobian.math.geometry.polygon_kernel._models import (
    KernelBoundaryIntersection,
    KernelPolygon,
    OrientedEdgeHalfPlane,
    PolygonVertexTurn,
)


def _fraction_point(point: RationalPoint2D) -> tuple[Fraction, Fraction]:
    return point.x.as_fraction(), point.y.as_fraction()


def _wire_point(point: tuple[Fraction, Fraction]) -> RationalPoint2D:
    return RationalPoint2D(
        x=CanonicalRational.from_fraction(point[0]),
        y=CanonicalRational.from_fraction(point[1]),
    )


def oriented_half_planes(
    polygon: KernelPolygon,
) -> tuple[OrientedEdgeHalfPlane, ...]:
    """Return edge-derived inequalities whose nonnegative side is interior."""

    result: list[OrientedEdgeHalfPlane] = []
    points = tuple(_fraction_point(point) for point in polygon.points)
    for edge_index, ((x1, y1), (x2, y2)) in enumerate(
        zip(points, points[1:] + points[:1], strict=True)
    ):
        a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
        result.append(
            OrientedEdgeHalfPlane(
                edge_index=edge_index,
                a=CanonicalRational.from_fraction(a),
                b=CanonicalRational.from_fraction(b),
                c=CanonicalRational.from_fraction(c),
            )
        )
    return tuple(result)


def _fraction_half_plane(
    half_plane: OrientedEdgeHalfPlane,
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        half_plane.a.as_fraction(),
        half_plane.b.as_fraction(),
        half_plane.c.as_fraction(),
    )


def _evaluate(
    half_plane: tuple[Fraction, Fraction, Fraction],
    point: tuple[Fraction, Fraction],
) -> Fraction:
    a, b, c = half_plane
    return a * point[0] + b * point[1] + c


def _feasible_boundary_intersections(
    half_planes: tuple[OrientedEdgeHalfPlane, ...],
) -> tuple[tuple[Fraction, Fraction], ...]:
    coefficients = tuple(_fraction_half_plane(item) for item in half_planes)
    candidates: set[tuple[Fraction, Fraction]] = set()
    for first_index, (a, b, c) in enumerate(coefficients):
        for d, e, f in coefficients[first_index + 1 :]:
            determinant = a * e - d * b
            if determinant == 0:
                continue
            point = (
                (b * f - e * c) / determinant,
                (c * d - f * a) / determinant,
            )
            if all(_evaluate(half_plane, point) >= 0 for half_plane in coefficients):
                candidates.add(point)
    return tuple(sorted(candidates))


def _canonical_hull(
    points: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[RationalPoint2D, ...]:
    wire_points = tuple(_wire_point(point) for point in points)
    if len(wire_points) <= 1:
        return wire_points
    return convex_hull_points(PointSetRequest(points=wire_points)).points


def polygon_signed_area(points: tuple[RationalPoint2D, ...]) -> Fraction:
    """Return the exact signed shoelace area of one cyclic boundary."""

    if len(points) < 3:
        return Fraction()
    total = Fraction()
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += (
            point.x.as_fraction() * following.y.as_fraction()
            - point.y.as_fraction() * following.x.as_fraction()
        )
    return total / 2


def _turns(polygon: KernelPolygon) -> tuple[PolygonVertexTurn, ...]:
    points = tuple(_fraction_point(point) for point in polygon.points)
    rows: list[PolygonVertexTurn] = []
    for vertex_index, current in enumerate(points):
        previous = points[vertex_index - 1]
        following = points[(vertex_index + 1) % len(points)]
        cross = (current[0] - previous[0]) * (following[1] - current[1]) - (
            current[1] - previous[1]
        ) * (following[0] - current[0])
        rows.append(
            PolygonVertexTurn(
                vertex_index=vertex_index,
                cross=CanonicalRational.from_fraction(cross),
                kind=(
                    "CONVEX" if cross > 0 else "REFLEX" if cross < 0 else "COLLINEAR"
                ),
            )
        )
    return tuple(rows)


def _boundary_rows(
    points: tuple[RationalPoint2D, ...],
    half_planes: tuple[OrientedEdgeHalfPlane, ...],
) -> tuple[KernelBoundaryIntersection, ...]:
    coefficients = tuple(_fraction_half_plane(item) for item in half_planes)
    return tuple(
        KernelBoundaryIntersection(
            point=point,
            active_edge_indices=tuple(
                edge_index
                for edge_index, half_plane in enumerate(coefficients)
                if _evaluate(half_plane, _fraction_point(point)) == 0
            ),
        )
        for point in points
    )


@dataclass(frozen=True, slots=True)
class KernelData:
    convention: Literal["a*x+b*y+c>=0"]
    half_planes: tuple[OrientedEdgeHalfPlane, ...]
    vertex_turns: tuple[PolygonVertexTurn, ...]
    reflex_vertex_indices: tuple[int, ...]
    dimension: Literal["EMPTY", "POINT", "SEGMENT", "POLYGON"]
    boundary: tuple[KernelBoundaryIntersection, ...]
    convex_hull: GeometryConvexHullResult
    polygon_area: CanonicalRational
    kernel_area: CanonicalRational
    convex_hull_area: CanonicalRational
    kernel_to_polygon_area_ratio: CanonicalRational
    polygon_to_hull_area_ratio: CanonicalRational

    def as_tuple(self) -> tuple[object, ...]:
        return (
            self.convention,
            self.half_planes,
            self.vertex_turns,
            self.reflex_vertex_indices,
            self.dimension,
            self.boundary,
            self.convex_hull,
            self.polygon_area,
            self.kernel_area,
            self.convex_hull_area,
            self.kernel_to_polygon_area_ratio,
            self.polygon_to_hull_area_ratio,
        )


def compute_kernel_data(polygon: KernelPolygon) -> KernelData:
    """Compute the complete canonical kernel data used by producer and replay."""

    half_planes = oriented_half_planes(polygon)
    candidates = _feasible_boundary_intersections(half_planes)
    kernel_points = _canonical_hull(candidates)
    dimension: Literal["EMPTY", "POINT", "SEGMENT", "POLYGON"] = (
        "EMPTY"
        if not kernel_points
        else "POINT"
        if len(kernel_points) == 1
        else "SEGMENT"
        if len(kernel_points) == 2
        else "POLYGON"
    )
    turns = _turns(polygon)
    source_hull = convex_hull_points(PointSetRequest(points=polygon.points))
    polygon_area = polygon_signed_area(polygon.points)
    kernel_area = polygon_signed_area(kernel_points)
    hull_area = polygon_signed_area(source_hull.points)
    return KernelData(
        convention="a*x+b*y+c>=0",
        half_planes=half_planes,
        vertex_turns=turns,
        reflex_vertex_indices=tuple(
            row.vertex_index for row in turns if row.kind == "REFLEX"
        ),
        dimension=dimension,
        boundary=_boundary_rows(kernel_points, half_planes),
        convex_hull=source_hull,
        polygon_area=CanonicalRational.from_fraction(polygon_area),
        kernel_area=CanonicalRational.from_fraction(kernel_area),
        convex_hull_area=CanonicalRational.from_fraction(hull_area),
        kernel_to_polygon_area_ratio=CanonicalRational.from_fraction(
            kernel_area / polygon_area
        ),
        polygon_to_hull_area_ratio=CanonicalRational.from_fraction(
            polygon_area / hull_area
        ),
    )


__all__ = ["KernelData", "compute_kernel_data", "oriented_half_planes"]
