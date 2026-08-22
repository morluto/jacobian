"""Typed wire contracts for bounded rational lattice-polytope operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel

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

COORDINATE_DIGITS = 32_768
"""Per-component digit bound forwarded to the canonical rational validator."""

RepresentationName = Literal["vertices", "halfspaces"]
"""The exactly-one representation tag carried by requests and results."""


def _reject_out_of_budget_scan(
    lo: list[int],
    hi: list[int],
    dim: int,
) -> int:
    """Reject an integer bounding box outside the admitted scan budgets.

    Each axis may span at most ``MAX_BOUND_SPAN`` integer points and the
    product of the spans (the candidate points tested by the scan) must
    stay within the 10M-point scan budget.  Returns the total scan so
    callers can bound derived work against it.
    """
    total_scan = 1
    for k in range(dim):
        span = hi[k] - lo[k] + 1
        if span > MAX_BOUND_SPAN:
            raise ValueError(
                "the integer bounding box exceeds the "
                f"{MAX_BOUND_SPAN}-point per-axis span bound"
            )
        total_scan *= span
        if total_scan > 10_000_000:
            raise ValueError(
                "integer bounding box total scan exceeds the 10M-point budget"
            )
    return total_scan


def _floor_frac(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil_frac(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


class Vertex(StrictModel):
    """One rational vertex of a V-representation."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )


class Halfspace(StrictModel):
    """One rational half-space ``<a, x> <= b`` of an H-representation."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )
    offset: CanonicalRational = Field(
        description="The right-hand side ``b`` of ``<a, x> <= b``.",
    )

    @model_validator(mode="after")
    def require_nonzero_normal(self) -> Self:
        if all(c.as_fraction() == 0 for c in self.coefficients):
            raise ValueError("half-space coefficients must not all be zero")
        return self


class LatticePolytopeRequest(StrictModel):
    """A bounded rational polytope in exactly one representation."""

    vertices: tuple[Vertex, ...] | None = Field(
        default=None,
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
    halfspaces: tuple[Halfspace, ...] | None = Field(
        default=None,
        description=(
            "H-representation: the half-spaces ``<a_i, x> <= b_i``. "
            "Mutually exclusive with ``vertices``."
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
            raise ValueError(
                "exactly one of `vertices` or `halfspaces` must be provided"
            )
        if has_v:
            self._validate_vertices()
        else:
            self._validate_halfspaces()
        return self

    def _validate_vertices(self) -> None:
        assert self.vertices is not None  # for type checkers
        if len(self.vertices) < 1:
            raise ValueError("`vertices` must be non-empty")
        if len(self.vertices) > MAX_VERTICES:
            raise ValueError(f"`vertices` exceeds the {MAX_VERTICES}-vertex bound")
        for vertex in self.vertices:
            for coord in vertex.coordinates:
                require_bounded_rational(
                    coord, max_digits=COORDINATE_DIGITS, label="vertex coordinate"
                )
        dim = len(self.vertices[0].coordinates)
        if dim > self.dimension_bound:
            raise ValueError(
                f"dimension {dim} exceeds the dimension bound {self.dimension_bound}"
            )
        for vertex in self.vertices:
            if len(vertex.coordinates) != dim:
                raise ValueError("all vertices must share one dimension")
        self._validate_vertex_geometry()

    def _validate_vertex_geometry(self) -> None:
        """Admit the vertex geometry before any enumeration work.

        Enforces the facet-combination budget, full dimensionality, and
        the bounding-box scan budgets so that both operations fail closed
        as ``ValidationError`` instead of expanding unbounded work.
        """
        assert self.vertices is not None  # for type checkers
        verts_frac = [[c.as_fraction() for c in v.coordinates] for v in self.vertices]
        dim = len(verts_frac[0])
        # Facet-combination budget: C(n,d) subsets define candidate hyperplanes.
        if dim > 1:
            from math import comb

            from sympy import Matrix

            try:
                facet_combinations = comb(len(self.vertices), dim)
            except ValueError:
                facet_combinations = 10**18
            if facet_combinations > 700_000:
                raise ValueError(
                    "vertex facet enumeration exceeds the 700k-combination budget"
                )
            # Full-dimensionality admission: the exact facet enumeration assumes
            # the vertices affinely span the ambient dimension.  Reject
            # lower-dimensional V-representations here, regardless of vertex
            # count, so both operations fail closed as ValidationError instead
            # of scanning an unrelated bounding box.
            diffs = Matrix(
                [
                    [verts_frac[i][k] - verts_frac[0][k] for k in range(dim)]
                    for i in range(1, len(verts_frac))
                ]
            )
            if diffs.rank() < dim:
                raise ValueError(
                    "V-representation is not full-dimensional; "
                    "lower-dimensional hulls require exact handling"
                )
            # Membership work: every scanned candidate is tested against
            # every facet generated from the vertices, so the exact facet
            # count multiplies the bounding-box scan exactly as for
            # H-representations.
            from sympy import Rational as SympyRational

            from jacobian.math.lattice_polytopes._operations import (
                _facets_from_points,
            )

            facet_count = len(
                _facets_from_points(
                    [[SympyRational(c) for c in row] for row in verts_frac],
                    dim,
                )
            )
        else:
            facet_count = 1
        # Inclusive integer bounds of the vertex bounding box.
        lo = [min(v[k] for v in verts_frac) for k in range(dim)]
        hi = [max(v[k] for v in verts_frac) for k in range(dim)]
        lo_int = [_floor_frac(lo[k]) for k in range(dim)]
        hi_int = [_ceil_frac(hi[k]) for k in range(dim)]
        total_scan = _reject_out_of_budget_scan(lo_int, hi_int, dim)
        if total_scan * facet_count > MAX_FACET_TESTS:
            raise ValueError(
                "the vertex-hull scan evaluates up to total-scan times "
                "facet-count inequalities and exceeds the "
                f"{MAX_FACET_TESTS}-test budget; reduce point count or "
                "bounding-box size"
            )

    def _validate_halfspaces(self) -> None:
        assert self.halfspaces is not None  # for type checkers
        if len(self.halfspaces) < 1:
            raise ValueError("`halfspaces` must be non-empty")
        if len(self.halfspaces) > MAX_HALFSPACES:
            raise ValueError(
                f"`halfspaces` exceeds the {MAX_HALFSPACES}-half-space bound"
            )
        for halfspace in self.halfspaces:
            for coeff in halfspace.coefficients:
                require_bounded_rational(
                    coeff,
                    max_digits=COORDINATE_DIGITS,
                    label="half-space coefficient",
                )
            require_bounded_rational(
                halfspace.offset,
                max_digits=COORDINATE_DIGITS,
                label="half-space offset",
            )
        dim = len(self.halfspaces[0].coefficients)
        if dim > self.dimension_bound:
            raise ValueError(
                f"dimension {dim} exceeds the dimension bound {self.dimension_bound}"
            )
        for halfspace in self.halfspaces:
            if len(halfspace.coefficients) != dim:
                raise ValueError("all half-spaces must share one dimension")
        self._validate_halfspace_geometry()

    def _validate_halfspace_geometry(self) -> None:
        """Admit the half-space geometry before any enumeration work.

        Bounded-ness and non-emptiness are decided exactly here, and the
        vertex bounding box must satisfy the shared scan budgets, so an
        accepted request always describes a bounded, non-empty polytope
        whose scan stays inside the admitted membership-work budget.
        """
        assert self.halfspaces is not None  # for type checkers
        from jacobian.math.lattice_polytopes._operations import (
            _bounding_box,
            _dedupe_normalized_halfspaces,
            _is_bounded_h,
            _vertices_from_h_representation,
        )

        dim = len(self.halfspaces[0].coefficients)
        halfspaces_list: list[tuple[list[Fraction], Fraction]] = [
            (
                [c.as_fraction() for c in hs.coefficients],
                hs.offset.as_fraction(),
            )
            for hs in self.halfspaces
        ]
        # ``Fraction`` values are accepted by the sympy-based helpers
        # via implicit conversion.
        if not _is_bounded_h(halfspaces_list, dim):
            raise ValueError(
                "the H-representation is unbounded; lattice-point enumeration "
                "requires a bounded polytope"
            )
        verts, _ = _vertices_from_h_representation(halfspaces_list)
        if not verts:
            raise ValueError("the H-representation defines an empty polytope")
        # verts are list of SymPy Rationals
        box_dim = len(verts[0])
        lo_h, hi_h = _bounding_box(verts, box_dim)
        total_scan_h = _reject_out_of_budget_scan(lo_h, hi_h, box_dim)
        # Membership work: every candidate is tested against every distinct
        # facet inequality.  Normalization merges repeated inequalities so
        # duplicates cannot multiply the admitted work.
        unique_facets = len(_dedupe_normalized_halfspaces(halfspaces_list))
        if total_scan_h * unique_facets > MAX_FACET_TESTS:
            raise ValueError(
                "the scan evaluates up to total-scan times facet-count "
                f"inequalities and exceeds the {MAX_FACET_TESTS}-test budget"
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
                raise ValueError(
                    "lattice-point coordinate exceeds the "
                    f"{COORDINATE_DIGITS}-digit bound"
                )
        return self


class EnumerateLatticePointsResult(StrictModel):
    """The complete list of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1, le=MAX_DIMENSION)
    point_count: int = Field(ge=0)
    points: tuple[LatticePoint, ...]
    representation: RepresentationName

    @model_validator(mode="after")
    def require_complete_point_set(self) -> Self:
        if self.point_count != len(self.points):
            raise ValueError(
                "point_count must equal the number of returned lattice points"
            )
        seen = {point.coordinates for point in self.points}
        if len(seen) != len(self.points):
            raise ValueError("enumeration must not repeat a lattice point")
        for point in self.points:
            if len(point.coordinates) != self.dimension:
                raise ValueError(
                    "every lattice point must carry exactly `dimension` coordinates"
                )
        return self


class CountLatticePointsResult(StrictModel):
    """The number of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1, le=MAX_DIMENSION)
    point_count: int = Field(ge=0)
    representation: RepresentationName


class EnumerateLatticePointsRequest(LatticePolytopeRequest):
    """Enumeration admission: the serialized result must fit the output limits.

    The exact lattice-point count is computed during request validation
    (bounded by the admitted scan budget); an accepted enumerate request
    therefore always materializes within the point cap and the 10 MiB
    canonical JSON output limit instead of failing after acceptance.
    """

    @model_validator(mode="after")
    def require_enumeration_artifact_fits(self) -> Self:
        from jacobian.math.lattice_polytopes._operations import (
            enumeration_output_admission,
        )

        enumeration_output_admission(self)
        return self


__all__ = [
    "MAX_BOUND_SPAN",
    "MAX_DIMENSION",
    "MAX_FACET_TESTS",
    "MAX_HALFSPACES",
    "MAX_LATTICE_POINTS",
    "MAX_VERTICES",
    "CountLatticePointsResult",
    "EnumerateLatticePointsRequest",
    "EnumerateLatticePointsResult",
    "Halfspace",
    "LatticePoint",
    "LatticePolytopeRequest",
    "RepresentationName",
    "Vertex",
]
