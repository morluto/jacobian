"""Typed contracts for exact rational polygon visibility kernels."""

from __future__ import annotations

from math import comb
from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.math.geometry._models import (
    GeometryConvexHullResult,
    PolygonRequest,
    RationalPoint2D,
    _is_simple_ring,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


MAX_KERNEL_SOURCE_VERTICES = 64
MAX_KERNEL_COORDINATE_DIGITS = 64
MAX_HALF_PLANE_COEFFICIENT_DIGITS = 1_024
MAX_INTERSECTION_COMPONENT_DIGITS = 2_056
MAX_KERNEL_FEASIBILITY_WORK = 500_000_000
MAX_KERNEL_RESULT_CHARS = 500_000


def _estimate_visibility_kernel_result_characters(
    vertex_count: int,
    max_coordinate_digits: int,
    coefficient_digits: int,
    intersection_digits: int,
) -> int:
    """Conservatively reserve the complete canonical kernel result.

    The result retains the source polygon, one half-plane and one turn per
    source vertex, and no more than one kernel-boundary point per edge.  The
    four rational components of every retained point dominate the variable
    portion; the remaining term covers labels, indices, arrays, and the
    scalar area profile.  Keeping this calculation pure makes the admission
    proof directly testable against produced canonical JSON.
    """

    return vertex_count * (
        16 * max_coordinate_digits
        + 6 * coefficient_digits
        + 8 * intersection_digits
        + 4 * vertex_count
        + 400
    )


class KernelPolygon(PolygonRequest):
    """Operation-local bounded view of one simple CCW rational polygon.

    The wire shape is exactly ``PolygonRequest``. The additional validation is
    the visibility-kernel operation's execution envelope, not a second polygon
    representation.
    """

    model_config = ConfigDict(from_attributes=True)

    points: tuple[RationalPoint2D, ...] = Field(
        min_length=3,
        max_length=MAX_KERNEL_SOURCE_VERTICES,
        description=(
            "Distinct cyclic vertices of one simple counterclockwise rational "
            f"polygon; coordinate components have at most "
            f"{MAX_KERNEL_COORDINATE_DIGITS} digits."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_simple_counterclockwise_polygon(self) -> Self:
        max_coordinate_digits = max(
            canonical_rational_component_digits(component)
            for point in self.points
            for component in (point.x, point.y)
        )
        if max_coordinate_digits > MAX_KERNEL_COORDINATE_DIGITS:
            raise _validation_error(
                "polygon_coordinates_exceed_f_max_kernel",
                "polygon coordinates exceed the "
                f"{MAX_KERNEL_COORDINATE_DIGITS}-digit visibility-kernel bound",
            )

        # Constructing the n oriented edge equations is the only arithmetic
        # done before the pairwise expansion. Their actual normalized height
        # makes the following work/output admission representation-sensitive.
        from jacobian.math.geometry.polygon_kernel._kernel import (
            oriented_half_planes,
            polygon_signed_area,
        )

        half_planes = oriented_half_planes(self)
        coefficient_digits = max(
            canonical_rational_component_digits(value)
            for half_plane in half_planes
            for value in (half_plane.a, half_plane.b, half_plane.c)
        )
        if coefficient_digits > MAX_HALF_PLANE_COEFFICIENT_DIGITS:
            raise _validation_error(
                "oriented_half_plane_coefficients_exceed_f",
                "oriented half-plane coefficients exceed the "
                f"{MAX_HALF_PLANE_COEFFICIENT_DIGITS}-digit bound",
            )
        # A 2x2 boundary intersection forms products and differences of
        # rational coefficients, then divides two such rationals. Eight input
        # component heights plus constant sign/carry slack bounds either
        # reduced output component.
        intersection_digits = 8 * coefficient_digits + 8
        if intersection_digits > MAX_INTERSECTION_COMPONENT_DIGITS:
            raise _validation_error(
                "a_boundary_line_intersection_exceed_f",
                "a boundary-line intersection can exceed the "
                f"{MAX_INTERSECTION_COMPONENT_DIGITS}-digit component bound",
            )

        vertex_count = len(self.points)
        # The result retains the source once, n half-planes and turns, and at
        # most n hull/kernel vertices. The formula charges four rational
        # components per retained point plus JSON/index overhead.
        estimated_result_chars = _estimate_visibility_kernel_result_characters(
            vertex_count,
            max_coordinate_digits,
            coefficient_digits,
            intersection_digits,
        )
        if estimated_result_chars > MAX_KERNEL_RESULT_CHARS:
            raise _validation_error(
                "visibility_kernel_result_require_f_estimated",
                "visibility-kernel result can require "
                f"{estimated_result_chars} characters, exceeding the "
                f"{MAX_KERNEL_RESULT_CHARS}-character bound",
            )

        pair_count = comb(vertex_count, 2)
        # Every boundary-line pair may produce one exact point which is checked
        # against every half-plane. Integer/Fraction multiplication at height h
        # is conservatively charged h^2 units.
        feasibility_work = (
            pair_count * vertex_count * coefficient_digits * coefficient_digits
        )
        if feasibility_work > MAX_KERNEL_FEASIBILITY_WORK:
            raise _validation_error(
                "visibility_kernel_feasibility_work_f_c",
                "visibility-kernel feasibility work "
                f"C({vertex_count},2)*{vertex_count}*"
                f"{coefficient_digits}^2={feasibility_work} exceeds "
                f"{MAX_KERNEL_FEASIBILITY_WORK}",
            )

        if not _is_simple_ring(self.points):
            raise _validation_error(
                "visibility_kernel_input_a_simple_polygon",
                "visibility-kernel input must be a simple polygon",
            )
        if polygon_signed_area(self.points) <= 0:
            raise _validation_error(
                "visibility_kernel_vertices_use_counterclockwise_cyclic",
                "visibility-kernel vertices must use counterclockwise cyclic order",
            )
        return self


class PolygonKernelRequest(StrictModel):
    """Reconstruct one simple CCW polygon's exact visibility kernel."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Exact visibility-kernel reconstruction for a simple CCW "
                "rational polygon. Admission bounds C(n,2)*n boundary-point "
                "feasibility checks, coefficient/intersection digit growth, "
                f"and a {MAX_KERNEL_RESULT_CHARS}-character result estimate "
                "before pairwise boundary expansion."
            ),
            "feasibility_work_bound": MAX_KERNEL_FEASIBILITY_WORK,
            "result_character_bound": MAX_KERNEL_RESULT_CHARS,
        }
    )

    polygon: KernelPolygon = Field(
        description=(
            "One simple counterclockwise polygon. Edge i is directed from "
            "vertex i to vertex i+1 cyclically, and its closed interior side "
            "is the left half-plane a*x+b*y+c >= 0."
        )
    )


class OrientedEdgeHalfPlane(StrictModel):
    """Exact edge-derived coefficients for one closed interior half-plane."""

    edge_index: StrictInt = Field(ge=0, lt=MAX_KERNEL_SOURCE_VERTICES)
    a: CanonicalRational
    b: CanonicalRational
    c: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero_normal(self) -> Self:
        if self.a.as_fraction() == 0 and self.b.as_fraction() == 0:
            raise _validation_error(
                "an_oriented_half_plane_a_nonzero",
                "an oriented half-plane must have a nonzero normal",
            )
        return self


class PolygonVertexTurn(StrictModel):
    """The exact centered orientation cross at one source vertex."""

    vertex_index: StrictInt = Field(ge=0, lt=MAX_KERNEL_SOURCE_VERTICES)
    cross: CanonicalRational
    kind: Literal["CONVEX", "COLLINEAR", "REFLEX"]

    @model_validator(mode="after")
    def bind_turn_kind(self) -> Self:
        value = self.cross.as_fraction()
        expected = "CONVEX" if value > 0 else "REFLEX" if value < 0 else "COLLINEAR"
        if self.kind != expected:
            raise _validation_error(
                "turn_kind_sign_exact_cross",
                "turn kind must match the sign of its exact cross",
            )
        return self


class KernelBoundaryIntersection(StrictModel):
    """One canonical kernel boundary point and all tight source edges there."""

    point: RationalPoint2D
    active_edge_indices: tuple[StrictInt, ...] = Field(
        min_length=2,
        max_length=MAX_KERNEL_SOURCE_VERTICES,
    )

    @model_validator(mode="after")
    def require_canonical_active_edges(self) -> Self:
        if self.active_edge_indices != tuple(sorted(set(self.active_edge_indices))):
            raise _validation_error(
                "active_edge_indices_distinct_sorted",
                "active edge indices must be distinct and sorted",
            )
        return self


class PolygonKernelResult(StrictModel):
    """Source-bound exact half-plane reconstruction and rational area profile."""

    polygon: KernelPolygon
    interior_half_plane_convention: Literal["a*x+b*y+c>=0"]
    half_planes: tuple[OrientedEdgeHalfPlane, ...] = Field(
        min_length=3,
        max_length=MAX_KERNEL_SOURCE_VERTICES,
    )
    vertex_turns: tuple[PolygonVertexTurn, ...] = Field(
        min_length=3,
        max_length=MAX_KERNEL_SOURCE_VERTICES,
    )
    reflex_vertex_indices: tuple[StrictInt, ...] = Field(
        max_length=MAX_KERNEL_SOURCE_VERTICES
    )
    kernel_dimension: Literal["EMPTY", "POINT", "SEGMENT", "POLYGON"]
    kernel_boundary: tuple[KernelBoundaryIntersection, ...] = Field(
        max_length=MAX_KERNEL_SOURCE_VERTICES
    )
    convex_hull: GeometryConvexHullResult
    polygon_area: CanonicalRational
    kernel_area: CanonicalRational
    convex_hull_area: CanonicalRational
    kernel_to_polygon_area_ratio: CanonicalRational
    polygon_to_hull_area_ratio: CanonicalRational

    @model_validator(mode="after")
    def bind_result_to_source_polygon(self) -> Self:
        from jacobian.math.geometry.polygon_kernel._kernel import compute_kernel_data

        expected = compute_kernel_data(self.polygon)
        actual = (
            self.interior_half_plane_convention,
            self.half_planes,
            self.vertex_turns,
            self.reflex_vertex_indices,
            self.kernel_dimension,
            self.kernel_boundary,
            self.convex_hull,
            self.polygon_area,
            self.kernel_area,
            self.convex_hull_area,
            self.kernel_to_polygon_area_ratio,
            self.polygon_to_hull_area_ratio,
        )
        if actual != expected.as_tuple():
            raise _validation_error(
                "visibility_kernel_result_retained_source_polygon",
                "visibility-kernel result does not match the retained source polygon",
            )
        return self


__all__ = [
    "MAX_HALF_PLANE_COEFFICIENT_DIGITS",
    "MAX_INTERSECTION_COMPONENT_DIGITS",
    "MAX_KERNEL_COORDINATE_DIGITS",
    "MAX_KERNEL_FEASIBILITY_WORK",
    "MAX_KERNEL_RESULT_CHARS",
    "MAX_KERNEL_SOURCE_VERTICES",
    "KernelBoundaryIntersection",
    "KernelPolygon",
    "OrientedEdgeHalfPlane",
    "PolygonKernelRequest",
    "PolygonKernelResult",
    "PolygonVertexTurn",
]
