"""Typed wire contracts for bounded rational lattice-polytope operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    ExactInteger,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry.polytopes.values import Halfspace as RationalHalfspace
from jacobian.math.geometry.polytopes.values import Vertex as RationalVertex
from jacobian.math.polynomials.values import RationalPolynomial

MAX_DIMENSION = 4
"""Absolute upper bound on the ambient dimension of a polytope.

The lattice-point scan enumerates an integer bounding box, so the work
grows with the product of the per-axis spans. Dimension four is the
largest admitted ambient dimension; larger dimensions are rejected
before any enumeration begins.
"""

MAX_VERTICES = 64
"""Absolute upper bound on the number of vertices in a V-representation."""

MAX_HALFSPACES = 64
"""Absolute upper bound on the number of half-spaces in an H-representation."""

MAX_BOUND_SPAN = 10_000
"""Absolute upper bound on the integer span of the polytope in any axis.

The scan walks the full integer bounding box, so each axis may span at
most this many integer points. Together with ``MAX_DIMENSION`` this
bounds the total number of candidate integer points that are tested.
"""

MAX_LATTICE_POINTS = 1_000_000
"""Absolute upper bound on the number of returned lattice points.

``enumerate`` fails closed with a budget error before materializing more
lattice points than this. ``count`` returns the small exact integer
answer and therefore keeps scanning to the admitted bounding-box budget
(the 10M-candidate scan bound) instead of enforcing this cap; its result
is a single count, not a materialized list.
"""

MAX_FACET_TESTS = 100_000_000
"""Absolute upper bound on exact membership evaluations during one scan.

Each scanned candidate point is tested against every facet inequality,
and an all-interior box reaches every facet for every candidate, so the
membership work of one accepted request is conservatively bounded by
``total_scan * facet_count``.  Half-spaces are normalized and deduplicated
before this product is formed, so repeated inequalities never multiply
the work.  Requests whose deduplicated facet count times their integer
bounding-box scan exceeds this budget are rejected at validation.
"""

MAX_FACET_COMBINATIONS = 700_000
"""Maximum V-representation facet candidates admitted by one lattice scan."""

MAX_TOTAL_SCAN = 10_000_000
"""Absolute upper bound on candidate points tested by one admitted scan.

Every accepted request's integer bounding box stays within this many
integer candidates, so neither operation can ever observe more lattice
points than this; the count result is constrained to the same maximum.
"""

MAX_EHRHART_DILATIONS = 32
"""Maximum number of dilation values retained by Ehrhart interpolation."""

COORDINATE_DIGITS = 32_768
"""Per-component digit bound forwarded to the canonical rational validator."""


RepresentationName = Literal["vertices", "halfspaces"]
"""The exactly-one representation tag carried by requests and results."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by lattice-polytope contracts."""

    return PydanticCustomError(f"lattice_polytope.{reason}", message)


class LatticePolytopeRequest(StrictModel):
    """A bounded rational polytope in exactly one representation."""

    vertices: tuple[RationalVertex, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "V-representation: the vertices of the convex hull.  The "
            "vertices must affinely span the ambient dimension "
            "(full-dimensional hull); other lower-dimensional "
            "V-representations are rejected.  The supported exception is a "
            "one-dimensional input: every 1-D vertex family, including a "
            "single point, is accepted and processed exactly.  Mutually "
            "exclusive with ``halfspaces``."
        ),
    )
    halfspaces: tuple[RationalHalfspace, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_HALFSPACES,
        description=(
            "H-representation: the half-spaces ``<a_i, x> <= b_i``, each "
            "with at least one nonzero coefficient (constant rows are "
            "rejected).  Mutually exclusive with ``vertices``.  A bounded "
            "but empty system is valid and yields the exact empty result "
            "(count zero, no points)."
        ),
    )
    dimension_bound: int = Field(
        default=MAX_DIMENSION,
        le=MAX_DIMENSION,
        ge=1,
        description=(
            "Upper bound on the ambient dimension; the request is rejected "
            "when the representation implies a larger dimension."
        ),
    )

    @model_validator(mode="after")
    def validate_representation(self) -> Self:
        has_v = self.vertices is not None
        has_h = self.halfspaces is not None
        if has_v == has_h:
            raise _validation_error(
                "representation_not_exclusive",
                "exactly one of `vertices` or `halfspaces` must be provided",
            )
        if has_v:
            self._validate_vertices()
        else:
            self._validate_halfspaces()
        return self

    def _validate_vertices(self) -> None:
        assert self.vertices is not None  # for type checkers
        if len(self.vertices) < 1:
            raise _validation_error("vertices_empty", "`vertices` must be non-empty")
        if len(self.vertices) > MAX_VERTICES:
            raise _validation_error(
                "vertices_too_many",
                f"`vertices` exceeds the {MAX_VERTICES}-vertex bound",
            )
        for vertex in self.vertices:
            for coord in vertex.coordinates:
                try:
                    require_bounded_rational(
                        coord, max_digits=COORDINATE_DIGITS, label="vertex coordinate"
                    )
                except ValueError as exc:
                    raise _validation_error(
                        "coordinate_out_of_bounds", str(exc)
                    ) from exc
        dim = len(self.vertices[0].coordinates)
        if dim > self.dimension_bound:
            raise _validation_error(
                "dimension_exceeds_bound",
                f"dimension {dim} exceeds the dimension bound {self.dimension_bound}",
            )
        for vertex in self.vertices:
            if len(vertex.coordinates) != dim:
                raise _validation_error(
                    "vertices_dimension_mismatch",
                    "all vertices must share one dimension",
                )

    def _validate_halfspaces(self) -> None:
        assert self.halfspaces is not None  # for type checkers
        if len(self.halfspaces) < 1:
            raise _validation_error(
                "halfspaces_empty", "`halfspaces` must be non-empty"
            )
        if len(self.halfspaces) > MAX_HALFSPACES:
            raise _validation_error(
                "halfspaces_too_many",
                f"`halfspaces` exceeds the {MAX_HALFSPACES}-half-space bound",
            )
        for halfspace in self.halfspaces:
            for coeff in halfspace.coefficients:
                try:
                    require_bounded_rational(
                        coeff,
                        max_digits=COORDINATE_DIGITS,
                        label="half-space coefficient",
                    )
                except ValueError as exc:
                    raise _validation_error(
                        "coordinate_out_of_bounds", str(exc)
                    ) from exc
            try:
                require_bounded_rational(
                    halfspace.offset,
                    max_digits=COORDINATE_DIGITS,
                    label="half-space offset",
                )
            except ValueError as exc:
                raise _validation_error("coordinate_out_of_bounds", str(exc)) from exc
        dim = len(self.halfspaces[0].coefficients)
        if dim > self.dimension_bound:
            raise _validation_error(
                "dimension_exceeds_bound",
                f"dimension {dim} exceeds the dimension bound {self.dimension_bound}",
            )
        for halfspace in self.halfspaces:
            if len(halfspace.coefficients) != dim:
                raise _validation_error(
                    "halfspaces_dimension_mismatch",
                    "all half-spaces must share one dimension",
                )

    def dimension(self) -> int:
        """Return the ambient dimension implied by the chosen representation."""
        if self.vertices is not None:
            return len(self.vertices[0].coordinates)
        assert self.halfspaces is not None
        return len(self.halfspaces[0].coefficients)


class LatticePoint(StrictModel):
    """One lattice point, as a tuple of canonical integers."""

    coordinates: tuple[ExactInteger, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )

    @classmethod
    def _from_kernel(cls, coordinates: tuple[ExactInteger, ...]) -> Self:
        """Construct a lattice point from trusted canonical kernel output."""
        return cls.model_construct(coordinates=coordinates)


class EnumerateLatticePointsResult(StrictModel):
    """The complete list of lattice points inside a bounded rational polytope.

    The artifact is capped at ``MAX_LATTICE_POINTS`` points, the same
    materialization bound admission enforces on every accepted enumerate
    request, so the serialized result cannot represent an enumeration no
    admitted request can produce.
    """

    dimension: int = Field(ge=1, le=MAX_DIMENSION)
    point_count: int = Field(ge=0, le=MAX_LATTICE_POINTS)
    points: tuple[LatticePoint, ...] = Field(max_length=MAX_LATTICE_POINTS)
    representation: RepresentationName

    @model_validator(mode="after")
    def require_complete_point_set(self) -> Self:
        if self.point_count != len(self.points):
            raise _validation_error(
                "point_count_mismatch",
                "point_count must equal the number of returned lattice points",
            )
        seen = {point.coordinates for point in self.points}
        if len(seen) != len(self.points):
            raise _validation_error(
                "duplicate_lattice_point", "enumeration must not repeat a lattice point"
            )
        for point in self.points:
            if len(point.coordinates) != self.dimension:
                raise _validation_error(
                    "point_dimension_mismatch",
                    "every lattice point must carry exactly `dimension` coordinates",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        dimension: int,
        points: tuple[LatticePoint, ...],
        representation: RepresentationName,
    ) -> Self:
        """Construct a result from the owner's already-admitted scan output."""
        return cls.model_construct(
            dimension=dimension,
            point_count=len(points),
            points=points,
            representation=representation,
        )


class CountLatticePointsResult(StrictModel):
    """The number of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1, le=MAX_DIMENSION)
    point_count: int = Field(ge=0, le=MAX_TOTAL_SCAN)
    representation: RepresentationName

    @classmethod
    def _from_kernel(
        cls,
        *,
        dimension: int,
        point_count: int,
        representation: RepresentationName,
    ) -> Self:
        """Construct a result from the owner's already-admitted scan output."""
        return cls.model_construct(
            dimension=dimension,
            point_count=point_count,
            representation=representation,
        )


class EnumerateLatticePointsRequest(LatticePolytopeRequest):
    """Wire request for enumeration; execution admission happens in the operation."""


def _require_ehrhart_request_shape(
    vertices: tuple[RationalVertex, ...], degree_bound: int, max_dilation: int
) -> int:
    """Check cheap request shape and scalar bounds before native admission."""

    dimension = _require_ehrhart_vertex_structure(vertices)
    if not 1 <= len(vertices) <= MAX_VERTICES:
        raise _validation_error(
            "ehrhart_vertex_count", "vertex count exceeds the admitted range"
        )
    if type(degree_bound) is not int or not 1 <= degree_bound <= MAX_DIMENSION:
        raise _validation_error(
            "ehrhart_degree_range", "degree_bound exceeds the admitted range"
        )
    if type(max_dilation) is not int or not 1 <= max_dilation <= MAX_EHRHART_DILATIONS:
        raise _validation_error(
            "ehrhart_dilation_range", "max_dilation exceeds the admitted range"
        )
    if degree_bound < dimension:
        raise _validation_error(
            "ehrhart_degree_bound", "degree_bound must cover the polytope dimension"
        )
    if max_dilation < degree_bound:
        raise _validation_error(
            "ehrhart_dilation_range",
            "max_dilation must provide degree_bound + 1 evaluations",
        )
    return dimension


def _require_ehrhart_vertex_structure(
    vertices: tuple[RationalVertex, ...],
) -> int:
    """Check source vertex shape and scalar representation without rank work."""

    if not 1 <= len(vertices) <= MAX_VERTICES:
        raise _validation_error(
            "ehrhart_vertex_count", "vertex count exceeds the admitted range"
        )
    dimensions = {len(vertex.coordinates) for vertex in vertices}
    if len(dimensions) != 1:
        raise _validation_error(
            "ehrhart_vertex_dimension", "all vertices must share one dimension"
        )
    dimension = next(iter(dimensions))
    if not 1 <= dimension <= MAX_DIMENSION:
        raise _validation_error(
            "ehrhart_dimension_exceeded",
            "Ehrhart dimension exceeds the supported bound",
        )
    if any(
        coordinate.den != 1 for vertex in vertices for coordinate in vertex.coordinates
    ):
        raise _validation_error(
            "ehrhart_requires_integral_vertices",
            "Ehrhart polynomial recovery currently requires integral vertices",
        )
    return dimension


def _require_ehrhart_full_dimensional(
    vertices: tuple[RationalVertex, ...],
) -> int:
    """Require the source vertices to affinely span their ambient dimension."""

    dimension = _require_ehrhart_vertex_structure(vertices)
    base = vertices[0].coordinates
    rows = [
        [
            vertex.coordinates[index].as_fraction() - base[index].as_fraction()
            for index in range(dimension)
        ]
        for vertex in vertices[1:]
    ]
    rank = 0
    for column in range(dimension):
        pivot = next(
            (row for row in range(rank, len(rows)) if rows[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row != rank and rows[row][column] != 0:
                factor = rows[row][column]
                rows[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[row], rows[rank], strict=True)
                ]
        rank += 1
        if rank == len(rows):
            break
    if rank < dimension:
        raise _validation_error(
            "ehrhart_not_full_dimensional",
            "Ehrhart source vertices must affinely span their ambient dimension",
        )
    return dimension


def _admit_ehrhart_scaled_coordinates(
    vertices: tuple[RationalVertex, ...], max_dilation: int
) -> None:
    """Admit maximum dilated height once at the native execution boundary."""

    scaled_digits = 0
    for vertex in vertices:
        for coordinate in vertex.coordinates:
            scaled_digits = max(
                scaled_digits,
                len(format_canonical_integer(abs(coordinate.num * max_dilation))),
            )
    if scaled_digits > MAX_CANONICAL_INTEGER_DIGITS:
        raise _validation_error(
            "ehrhart_scaled_coordinate",
            "a maximum-dilation coordinate exceeds the canonical integer bound",
        )


def require_ehrhart_source(
    vertices: tuple[RationalVertex, ...], degree_bound: int, max_dilation: int
) -> None:
    """Admit one native Ehrhart source, including affine dimension."""

    _require_ehrhart_request_shape(vertices, degree_bound, max_dilation)
    _admit_ehrhart_scaled_coordinates(vertices, max_dilation)
    _require_ehrhart_full_dimensional(vertices)


class EhrhartRequest(StrictModel):
    """Recover an Ehrhart polynomial for a bounded integral V-polytope."""

    vertices: tuple[RationalVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "Integral vertices of a full-dimensional bounded V-polytope. "
            "Integral vertices are required because rational polytopes have "
            "quasi-polynomial Ehrhart counts."
        ),
    )
    degree_bound: int = Field(ge=1, le=MAX_DIMENSION)
    max_dilation: int = Field(default=MAX_DIMENSION, ge=1, le=MAX_EHRHART_DILATIONS)

    @model_validator(mode="after")
    def require_integral_vertices_and_range(self) -> Self:
        _require_ehrhart_request_shape(
            self.vertices, self.degree_bound, self.max_dilation
        )
        return self


class EhrhartResult(StrictModel):
    """A source-bound exact Ehrhart polynomial and its retained count table."""

    vertices: tuple[RationalVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
    )

    dimension: int = Field(ge=1, le=MAX_DIMENSION)
    degree_bound: int = Field(ge=1, le=MAX_DIMENSION)
    max_dilation: int = Field(ge=1, le=MAX_EHRHART_DILATIONS)
    counts: tuple[tuple[int, ExactInteger], ...] = Field(
        min_length=2, max_length=MAX_EHRHART_DILATIONS + 1
    )
    polynomial: RationalPolynomial

    @model_validator(mode="after")
    def require_result_shapes(self) -> Self:
        dimension = _require_ehrhart_request_shape(
            self.vertices, self.degree_bound, self.max_dilation
        )
        if self.dimension != dimension:
            raise _validation_error(
                "ehrhart_dimension_shape",
                "result dimension must match the source vertex dimension",
            )
        if self.max_dilation < self.degree_bound:
            raise _validation_error(
                "ehrhart_dilation_range",
                "max_dilation must provide degree_bound + 1 evaluations",
            )
        if len(self.counts) != self.max_dilation + 1:
            raise _validation_error(
                "ehrhart_count_shape",
                "counts must contain every dilation from zero through max_dilation",
            )
        if tuple(t for t, _count in self.counts) != tuple(range(self.max_dilation + 1)):
            raise _validation_error(
                "ehrhart_count_dilation",
                "dilation values must be the complete range from zero",
            )
        if any(count < 0 for _dilation, count in self.counts):
            raise _validation_error(
                "ehrhart_count_value", "lattice counts must be nonnegative"
            )
        if self.polynomial.variables != ("t",):
            raise _validation_error(
                "ehrhart_polynomial_axis",
                "Ehrhart polynomials must use the canonical `t` axis",
            )
        if any(
            term.exponents[0] > self.degree_bound
            for term in self.polynomial.polynomial.terms
        ):
            raise _validation_error(
                "ehrhart_polynomial_degree",
                "polynomial degree must not exceed degree_bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        vertices: tuple[RationalVertex, ...],
        dimension: int,
        degree_bound: int,
        max_dilation: int,
        counts: tuple[tuple[int, ExactInteger], ...],
        polynomial: RationalPolynomial,
    ) -> Self:
        """Construct trusted output after native admission and counting."""

        return cls.model_construct(
            vertices=vertices,
            dimension=dimension,
            degree_bound=degree_bound,
            max_dilation=max_dilation,
            counts=counts,
            polynomial=polynomial,
        )


__all__ = [
    "MAX_BOUND_SPAN",
    "MAX_DIMENSION",
    "MAX_EHRHART_DILATIONS",
    "MAX_FACET_COMBINATIONS",
    "MAX_FACET_TESTS",
    "MAX_HALFSPACES",
    "MAX_LATTICE_POINTS",
    "MAX_TOTAL_SCAN",
    "MAX_VERTICES",
    "CountLatticePointsResult",
    "EhrhartRequest",
    "EhrhartResult",
    "EnumerateLatticePointsRequest",
    "EnumerateLatticePointsResult",
    "LatticePoint",
    "LatticePolytopeRequest",
    "RepresentationName",
]
