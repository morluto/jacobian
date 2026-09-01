"""Typed contracts for exact rational polygon visibility kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

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

if TYPE_CHECKING:
    from jacobian.math.geometry.polygon_kernel._kernel import KernelData


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by the geometry contracts."""

    return PydanticCustomError(f"geometry.{reason}", message)


MAX_KERNEL_SOURCE_VERTICES = 64
MAX_KERNEL_COORDINATE_DIGITS = 64
MAX_HALF_PLANE_COEFFICIENT_DIGITS = 1_024
MAX_INTERSECTION_COMPONENT_DIGITS = 2_056
MAX_KERNEL_FEASIBILITY_WORK = 500_000_000
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

        if not _is_simple_ring(self.points):
            raise _validation_error(
                "visibility_kernel_input_a_simple_polygon",
                "visibility-kernel input must be a simple polygon",
            )
        from jacobian.math.geometry.polygon_kernel._kernel import polygon_signed_area

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
                "feasibility checks and coefficient/intersection digit growth "
                "before pairwise boundary expansion."
            ),
            "feasibility_work_bound": MAX_KERNEL_FEASIBILITY_WORK,
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
    def require_structural_consistency(self) -> Self:
        vertex_count = len(self.polygon.points)
        if (
            len(self.half_planes) != vertex_count
            or len(self.vertex_turns) != vertex_count
        ):
            raise _validation_error(
                "visibility_kernel_result_source_row_count",
                "visibility-kernel half-planes and turns must have one row per source vertex",
            )
        if tuple(row.edge_index for row in self.half_planes) != tuple(
            range(vertex_count)
        ):
            raise _validation_error(
                "visibility_kernel_result_half_plane_indices",
                "visibility-kernel half-plane indices must be consecutive source indices",
            )
        if tuple(row.vertex_index for row in self.vertex_turns) != tuple(
            range(vertex_count)
        ):
            raise _validation_error(
                "visibility_kernel_result_turn_indices",
                "visibility-kernel turn indices must be consecutive source indices",
            )
        if self.reflex_vertex_indices != tuple(sorted(set(self.reflex_vertex_indices))):
            raise _validation_error(
                "visibility_kernel_result_reflex_indices",
                "visibility-kernel reflex indices must be distinct and sorted",
            )
        if any(
            index < 0 or index >= vertex_count for index in self.reflex_vertex_indices
        ):
            raise _validation_error(
                "visibility_kernel_result_reflex_indices",
                "visibility-kernel reflex indices must refer to source vertices",
            )
        expected_boundary_size = {
            "EMPTY": 0,
            "POINT": 1,
            "SEGMENT": 2,
        }.get(self.kernel_dimension)
        if expected_boundary_size is not None:
            valid_boundary_size = len(self.kernel_boundary) == expected_boundary_size
        else:
            valid_boundary_size = len(self.kernel_boundary) >= 3
        if not valid_boundary_size:
            raise _validation_error(
                "visibility_kernel_result_dimension_boundary",
                "visibility-kernel dimension must match its boundary cardinality",
            )
        return self

    @classmethod
    def _from_kernel(cls, polygon: KernelPolygon, *, data: KernelData) -> Self:
        """Build a result after the admitted kernel established its values."""

        return cls.model_construct(
            polygon=polygon,
            interior_half_plane_convention=data.convention,
            half_planes=data.half_planes,
            vertex_turns=data.vertex_turns,
            reflex_vertex_indices=data.reflex_vertex_indices,
            kernel_dimension=data.dimension,
            kernel_boundary=data.boundary,
            convex_hull=data.convex_hull,
            polygon_area=data.polygon_area,
            kernel_area=data.kernel_area,
            convex_hull_area=data.convex_hull_area,
            kernel_to_polygon_area_ratio=data.kernel_to_polygon_area_ratio,
            polygon_to_hull_area_ratio=data.polygon_to_hull_area_ratio,
        )


__all__ = [
    "MAX_HALF_PLANE_COEFFICIENT_DIGITS",
    "MAX_INTERSECTION_COMPONENT_DIGITS",
    "MAX_KERNEL_COORDINATE_DIGITS",
    "MAX_KERNEL_FEASIBILITY_WORK",
    "MAX_KERNEL_SOURCE_VERTICES",
    "KernelBoundaryIntersection",
    "KernelPolygon",
    "OrientedEdgeHalfPlane",
    "PolygonKernelRequest",
    "PolygonKernelResult",
    "PolygonVertexTurn",
]
