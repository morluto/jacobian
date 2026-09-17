"""Typed contracts for exact rational polygon visibility kernels."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._models import StrictModel
from jacobian.math.geometry._models import (
    GeometryConvexHullResult,
    RationalPoint2D,
    RationalPolygon2D,
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


class KernelPolygon(RationalPolygon2D):
    """Operation-local bounded view of one simple CCW rational polygon.

    The wire shape is exactly ``RationalPolygon2D``. The additional validation is
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
    def require_bounded_coordinates(self) -> Self:
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

    polygon: RationalPolygon2D
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
            polygon=RationalPolygon2D(points=polygon.points),
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


MAX_MEASURE_COMPARISONS = 16

MeasureSelector = Literal[
    "POLYGON_AREA",
    "KERNEL_AREA",
    "HULL_AREA",
    "POLYGON_PERIMETER",
    "HULL_PERIMETER",
    "KERNEL_PERIMETER",
    "KERNEL_TO_POLYGON_AREA_RATIO",
    "POLYGON_TO_HULL_AREA_RATIO",
]

MeasureOperator = Literal["LT", "LE", "EQ", "GE", "GT"]


class EdgeLengthEntry(StrictModel):
    """One exact cyclic edge length: its square always, its root when rational."""

    squared_length: CanonicalRational
    exact_length: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_length_shape(self) -> Self:
        if self.squared_length.as_fraction() < 0:
            raise _validation_error(
                "measure_edge_squared_nonnegative",
                "a squared edge length is nonnegative",
            )
        if self.exact_length is not None:
            if self.exact_length.as_fraction() < 0:
                raise _validation_error(
                    "measure_edge_length_nonnegative",
                    "an exact edge length is nonnegative",
                )
            if (
                self.exact_length.as_fraction() * self.exact_length.as_fraction()
                != self.squared_length.as_fraction()
            ):
                raise _validation_error(
                    "measure_edge_length_consistency",
                    "an exact edge length must square to its squared length",
                )
        return self


class RingMeasureProfile(StrictModel):
    """Exact edge lengths and perimeter of one cyclic boundary, if rational."""

    vertices: tuple[RationalPoint2D, ...] = Field(max_length=128)
    edges: tuple[EdgeLengthEntry, ...] = Field(max_length=128)
    perimeter: CanonicalRational | None = None

    @model_validator(mode="after")
    def require_ring_shape(self) -> Self:
        if len(self.edges) not in (0, len(self.vertices)):
            raise _validation_error(
                "measure_ring_edge_count",
                "a ring carries no edges or one edge per cyclic vertex",
            )
        if self.edges and any(entry.exact_length is None for entry in self.edges) != (
            self.perimeter is None
        ):
            raise _validation_error(
                "measure_ring_perimeter_agreement",
                "a perimeter is present exactly when every edge length is rational",
            )
        return self


class MeasureComparison(StrictModel):
    """One caller-specified exact comparison between two scalar measures."""

    left: MeasureSelector
    right: MeasureSelector
    operator: MeasureOperator
    expected_difference: CanonicalRational | None = None


class MeasureComparisonOutcome(StrictModel):
    """The exact difference of one comparison and whether it holds."""

    comparison: MeasureComparison
    difference: CanonicalRational
    holds: bool


class MeasureCertificateRequest(StrictModel):
    """Certify exact lengths, perimeters, and comparisons over a kernel result."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Exact measure certificate over a `PolygonKernelResult`: the "
                "kernel replays every relied-upon half-plane, boundary point, "
                "and area, then extends the profile with exact edge lengths, "
                "rational perimeters where every edge is rational, and "
                "caller-specified exact comparisons."
            )
        }
    )

    kernel: PolygonKernelResult = Field(
        description=(
            "The bound kernel result whose half-planes, boundary, hull, and "
            "areas are replayed before any measure is certified."
        ),
    )
    comparisons: tuple[MeasureComparison, ...] = Field(
        default=(),
        max_length=MAX_MEASURE_COMPARISONS,
        description=(
            "Caller-specified exact comparisons between scalar measures; "
            "each referenced measure must be present (rational)."
        ),
    )


class MeasureCertificateResult(StrictModel):
    """Exact measure profile bound to one replayed kernel result.

    ``polygon_measures``/``hull_measures`` ring the source polygon and its
    convex hull.  ``kernel_measures`` rings the kernel boundary when the
    kernel is a polygon, holds its two endpoints with an exact segment
    length when the kernel is a segment, holds its single point with a
    zero perimeter when the kernel is a point, and is empty when the
    kernel is empty.
    """

    kernel: PolygonKernelResult
    polygon_measures: RingMeasureProfile
    hull_measures: RingMeasureProfile
    kernel_measures: RingMeasureProfile
    kernel_segment_length: CanonicalRational | None = None
    comparisons: tuple[MeasureComparisonOutcome, ...] = Field(
        default=(), max_length=MAX_MEASURE_COMPARISONS
    )

    @model_validator(mode="after")
    def require_measure_rings(self) -> Self:
        if (
            tuple(self.polygon_measures.vertices) != self.kernel.polygon.points
            or tuple(self.hull_measures.vertices) != self.kernel.convex_hull.points
        ):
            raise _validation_error(
                "measure_ring_vertices",
                "measure rings must use the bound polygon and hull vertices",
            )
        boundary = tuple(row.point for row in self.kernel.kernel_boundary)
        dimension = self.kernel.kernel_dimension
        if dimension == "POLYGON":
            if (
                tuple(self.kernel_measures.vertices) != boundary
                or self.kernel_segment_length is not None
            ):
                raise _validation_error(
                    "measure_kernel_polygon_ring",
                    "a polygon kernel rings its boundary with no segment length",
                )
        elif dimension == "SEGMENT":
            if (
                tuple(self.kernel_measures.vertices) != boundary
                or self.kernel_measures.edges
                or self.kernel_measures.perimeter is not None
                or self.kernel_segment_length is None
            ):
                raise _validation_error(
                    "measure_kernel_segment_ring",
                    "a segment kernel holds its endpoints with one segment length",
                )
        elif dimension == "POINT":
            if (
                tuple(self.kernel_measures.vertices) != boundary
                or self.kernel_measures.edges
                or self.kernel_measures.perimeter is None
                or self.kernel_measures.perimeter.as_fraction() != 0
                or self.kernel_segment_length is not None
            ):
                raise _validation_error(
                    "measure_kernel_point_ring",
                    "a point kernel holds its point with a zero perimeter",
                )
        elif (
            tuple(self.kernel_measures.vertices) != ()
            or self.kernel_measures.edges
            or self.kernel_measures.perimeter is not None
            or self.kernel_segment_length is not None
        ):
            raise _validation_error(
                "measure_kernel_empty_ring",
                "an empty kernel holds no ring measures",
            )
        for profile in (
            self.polygon_measures,
            self.hull_measures,
            self.kernel_measures,
        ):
            if len(profile.edges) not in (0, len(profile.vertices)):
                raise _validation_error(
                    "measure_ring_edge_count",
                    "a ring carries no edges or one edge per cyclic vertex",
                )
        return self

    @model_validator(mode="after")
    def require_measure_orientation(self) -> Self:
        if self.kernel.polygon_area.as_fraction() <= 0:
            raise _validation_error(
                "measure_polygon_area_positive",
                "the bound polygon must have positive (counterclockwise) area",
            )
        return self

    @model_validator(mode="after")
    def require_measure_scalars(self) -> Self:
        for profile in (
            self.polygon_measures,
            self.hull_measures,
            self.kernel_measures,
        ):
            if profile.edges and any(
                entry.exact_length is None for entry in profile.edges
            ) != (profile.perimeter is None):
                raise _validation_error(
                    "measure_ring_perimeter_agreement",
                    "a perimeter is present exactly when every edge length is rational",
                )
        scalars: dict[str, CanonicalRational | None] = _measure_scalars(self)
        for outcome in self.comparisons:
            left = scalars[outcome.comparison.left]
            right = scalars[outcome.comparison.right]
            if left is None or right is None:
                raise _validation_error(
                    "measure_comparison_available",
                    "every compared measure must be present (rational)",
                )
            difference, holds = _evaluate_comparison(
                left.as_fraction(),
                right.as_fraction(),
                outcome.comparison.operator,
                outcome.comparison.expected_difference.as_fraction()
                if outcome.comparison.expected_difference is not None
                else None,
            )
            if (
                CanonicalRational.from_fraction(difference) != outcome.difference
                or holds != outcome.holds
            ):
                raise _validation_error(
                    "measure_comparison_consistency",
                    "every comparison outcome must match its exact difference",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _measure_scalars(
    result: MeasureCertificateResult,
) -> dict[str, CanonicalRational | None]:
    """Project the result's scalar measures by selector name."""

    return {
        "POLYGON_AREA": result.kernel.polygon_area,
        "KERNEL_AREA": result.kernel.kernel_area,
        "HULL_AREA": result.kernel.convex_hull_area,
        "POLYGON_PERIMETER": result.polygon_measures.perimeter,
        "HULL_PERIMETER": result.hull_measures.perimeter,
        "KERNEL_PERIMETER": result.kernel_measures.perimeter,
        "KERNEL_TO_POLYGON_AREA_RATIO": result.kernel.kernel_to_polygon_area_ratio,
        "POLYGON_TO_HULL_AREA_RATIO": result.kernel.polygon_to_hull_area_ratio,
    }


def _evaluate_comparison(
    left: Fraction,
    right: Fraction,
    operator: MeasureOperator,
    expected_difference: Fraction | None,
) -> tuple[Fraction, bool]:
    """Evaluate one exact scalar comparison."""

    difference = left - right
    holds = {
        "LT": difference < 0,
        "LE": difference <= 0,
        "EQ": difference == 0,
        "GE": difference >= 0,
        "GT": difference > 0,
    }[operator]
    if expected_difference is not None:
        holds = holds and difference == expected_difference
    return difference, holds


__all__ = [
    "MAX_HALF_PLANE_COEFFICIENT_DIGITS",
    "MAX_INTERSECTION_COMPONENT_DIGITS",
    "MAX_KERNEL_COORDINATE_DIGITS",
    "MAX_KERNEL_FEASIBILITY_WORK",
    "MAX_KERNEL_SOURCE_VERTICES",
    "MAX_MEASURE_COMPARISONS",
    "EdgeLengthEntry",
    "KernelBoundaryIntersection",
    "KernelPolygon",
    "MeasureCertificateRequest",
    "MeasureCertificateResult",
    "MeasureComparison",
    "MeasureComparisonOutcome",
    "MeasureOperator",
    "MeasureSelector",
    "OrientedEdgeHalfPlane",
    "PolygonKernelRequest",
    "PolygonKernelResult",
    "PolygonVertexTurn",
    "RingMeasureProfile",
]
