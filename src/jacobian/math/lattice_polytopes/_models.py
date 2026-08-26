"""Typed wire contracts for bounded rational lattice-polytope operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, PrivateAttr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalInteger,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.math.polytope.values import Halfspace as RationalHalfspace
from jacobian.math.polytope.values import Vertex as RationalVertex

AdmittedGeometry = tuple[list[tuple[tuple[int, ...], int]], list[int], list[int], int]
"""Trusted exact geometry retained on an admitted request."""

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

MAX_TOTAL_SCAN = 10_000_000
"""Absolute upper bound on candidate points tested by one admitted scan.

Every accepted request's integer bounding box stays within this many
integer candidates, so neither operation can ever observe more lattice
points than this; the count result is constrained to the same maximum.
"""

COORDINATE_DIGITS = 32_768
"""Per-component digit bound forwarded to the canonical rational validator."""


RepresentationName = Literal["vertices", "halfspaces"]
"""The exactly-one representation tag carried by requests and results."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by lattice-polytope contracts."""

    return PydanticCustomError(f"lattice_polytope.{reason}", message)


class LatticePolytopeRequest(StrictModel):
    """A bounded rational polytope in exactly one representation."""

    _geometry: AdmittedGeometry | None = PrivateAttr(default=None)

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
        self._validate_vertex_geometry()

    def _validate_vertex_geometry(self) -> None:
        """Admit the vertex geometry before any enumeration work.

        The admitted integer geometry is computed exactly once (the
        facet-combination budget, full dimensionality, and bounding-box
        scan budgets fail closed inside it) and memoized for admission
        and execution, so one accepted request never repeats the bounded
        facet-enumeration work.
        """
        try:
            membership_work = self._membership_work()
        except ValueError as exc:
            raise _validation_error("geometry_invalid", str(exc)) from exc
        if membership_work > MAX_FACET_TESTS:
            raise _validation_error(
                "geometry_work_exceeds_bound",
                "the vertex-hull scan evaluates up to total-scan times "
                "facet-count inequalities and exceeds the "
                f"{MAX_FACET_TESTS}-test budget; reduce point count or "
                "bounding-box size",
            )

    def admitted_geometry(self) -> AdmittedGeometry:
        """Return the admitted integer geometry, computing it once per request.

        Validation, artifact admission, and execution all need the exact
        facet inequalities and the integer bounding box; the first call
        computes them once and memoizes them on this request instance so
        no accepted request repeats the bounded facet-enumeration work.
        """
        geometry = self._geometry
        if geometry is None:
            from jacobian.math.lattice_polytopes._geometry_admission import (
                admitted_geometry,
            )

            geometry = admitted_geometry(self)
            self._geometry = geometry
        return geometry

    def _membership_work(self) -> int:
        """Return the admitted scan-times-facet-count membership bound."""
        facets, lo, hi, _dim = self.admitted_geometry()
        total_scan = 1
        for k in range(len(lo)):
            total_scan *= hi[k] - lo[k] + 1
        return total_scan * len(facets)

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
        self._validate_halfspace_geometry()

    def _validate_halfspace_geometry(self) -> None:
        """Admit the half-space geometry before any enumeration work.

        Bounded-ness is decided exactly inside the shared geometry
        computation, and membership work is bounded by distinct-facet
        count times bounding-box scan, so an accepted request always
        describes a bounded, possibly empty polytope whose scan stays
        inside the admitted work budget.
        """
        try:
            membership_work = self._membership_work()
        except ValueError as exc:
            raise _validation_error("geometry_invalid", str(exc)) from exc
        if membership_work > MAX_FACET_TESTS:
            raise _validation_error(
                "geometry_work_exceeds_bound",
                "the scan evaluates up to total-scan times facet-count "
                f"inequalities and exceeds the {MAX_FACET_TESTS}-test budget",
            )

    def dimension(self) -> int:
        """Return the ambient dimension implied by the chosen representation."""
        if self.vertices is not None:
            return len(self.vertices[0].coordinates)
        assert self.halfspaces is not None
        return len(self.halfspaces[0].coefficients)


class LatticePoint(StrictModel):
    """One lattice point, as a tuple of canonical integers."""

    coordinates: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )

    @model_validator(mode="after")
    def require_coordinate_digit_bound(self) -> Self:
        for coordinate in self.coordinates:
            if len(coordinate.lstrip("-")) > COORDINATE_DIGITS:
                raise _validation_error(
                    "lattice_point_coordinate_digit_bound",
                    "lattice-point coordinate exceeds the "
                    f"{COORDINATE_DIGITS}-digit bound",
                )
        return self

    @classmethod
    def _from_kernel(cls, coordinates: tuple[CanonicalInteger, ...]) -> Self:
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
    """Enumeration admission: the serialized result must fit the output limits.

    The exact lattice-point count is computed during request validation
    (bounded by the admitted scan budget); an accepted enumerate request
    therefore always materializes within the point cap and the 10 MiB
    canonical JSON output limit instead of failing after acceptance.
    """

    @model_validator(mode="after")
    def require_enumeration_artifact_fits(self) -> Self:
        from jacobian.math.lattice_polytopes._geometry_admission import (
            enumeration_output_admission,
        )

        try:
            enumeration_output_admission(self)
        except ValueError as exc:
            raise _validation_error("enumeration_artifact_invalid", str(exc)) from exc
        return self


__all__ = [
    "MAX_BOUND_SPAN",
    "MAX_DIMENSION",
    "MAX_FACET_TESTS",
    "MAX_HALFSPACES",
    "MAX_LATTICE_POINTS",
    "MAX_TOTAL_SCAN",
    "MAX_VERTICES",
    "CountLatticePointsResult",
    "EnumerateLatticePointsRequest",
    "EnumerateLatticePointsResult",
    "LatticePoint",
    "LatticePolytopeRequest",
    "RepresentationName",
]
