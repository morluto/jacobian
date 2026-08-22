"""Typed wire contracts for bounded rational lattice-polytope operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

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

COORDINATE_DIGITS = 32_768
"""Per-component digit bound forwarded to the canonical rational validator."""


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
            "(full-dimensional hull); lower-dimensional V-representations "
            "are rejected.  Mutually exclusive with ``halfspaces``."
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
        # Facet-combination budget: C(n,d) subsets define candidate hyperplanes
        if dim > 1:
            from math import comb as _comb

            try:
                facet_combinations = _comb(len(self.vertices), dim)
            except ValueError:
                facet_combinations = 10**18
            if facet_combinations > 700_000:
                raise ValueError(
                    "vertex facet enumeration exceeds the 700k-combination budget"
                )
        # Conservative bounding-box work budget: per-axis span and total
        # product must be bounded before any enumeration.  This is the
        # admission filter; the operation layer keeps the same safety check
        # but the request is rejected here as ValidationError.
        verts_frac = [[c.as_fraction() for c in v.coordinates] for v in self.vertices]
        # Full-dimensionality admission: the exact facet enumeration assumes
        # the vertices affinely span the ambient dimension.  Reject
        # lower-dimensional V-representations here, regardless of vertex
        # count, so both operations fail closed as ValidationError instead of
        # scanning an unrelated bounding box.
        if dim > 1:
            from sympy import Matrix

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
        # lo/hi inclusive integer bounds
        lo = [min(v[k] for v in verts_frac) for k in range(dim)]
        hi = [max(v[k] for v in verts_frac) for k in range(dim)]
        # Convert Fraction to integer bounds via floor/ceil
        import math as _math

        def _floor_frac(f: Fraction) -> int:
            return f.numerator // f.denominator

        def _ceil_frac(f: Fraction) -> int:
            return -((-f.numerator) // f.denominator)

        lo_int = [_floor_frac(lo[k]) for k in range(dim)]
        hi_int = [_ceil_frac(hi[k]) for k in range(dim)]
        for k in range(dim):
            span = hi_int[k] - lo_int[k] + 1
            if span > MAX_BOUND_SPAN:
                raise ValueError(
                    "the integer bounding box exceeds the "
                    f"{MAX_BOUND_SPAN}-point per-axis span bound"
                )
        total_scan = 1
        for k in range(dim):
            span = hi_int[k] - lo_int[k] + 1
            total_scan *= span
            if total_scan > 10_000_000:
                raise ValueError(
                    "integer bounding box total scan exceeds the 10M-point budget"
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
        # Bounded-ness and non-emptiness must be decided before any
        # exact enumeration.  These checks are the operation's domain
        # filter: an unbounded or empty H-polytope is not a valid
        # request, so it is rejected here as ``ValidationError`` rather
        # than as a host exception after acceptance.
        from jacobian.math.lattice_polytopes._operations import (
            _is_bounded_h,
            _vertices_from_h_representation,
        )

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
        # Bounding-box admission for H as well (product of spans bounded)
        from jacobian.math.lattice_polytopes._operations import _bounding_box

        # verts are list of SymPy Rationals
        dim_verts = len(verts[0])
        lo_h, hi_h = _bounding_box(verts, dim_verts)  # type: ignore[arg-type]
        for k in range(dim_verts):
            span = hi_h[k] - lo_h[k] + 1
            if span > MAX_BOUND_SPAN:
                raise ValueError(
                    "the integer bounding box exceeds the "
                    f"{MAX_BOUND_SPAN}-point per-axis span bound"
                )
        total_scan_h = 1
        for k in range(dim_verts):
            span = hi_h[k] - lo_h[k] + 1
            total_scan_h *= span
            if total_scan_h > 10_000_000:
                raise ValueError(
                    "integer bounding box total scan exceeds the 10M-point budget"
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
            if len(coordinate.lstrip("-")) > COORDINATE_DIGITS + 1:
                raise ValueError(
                    "lattice-point coordinate exceeds the "
                    f"{COORDINATE_DIGITS}-digit bound"
                )
        return self


class EnumerateLatticePointsResult(StrictModel):
    """The complete list of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1)
    point_count: int = Field(ge=0)
    points: tuple[LatticePoint, ...]
    representation: str

    @model_validator(mode="after")
    def require_count_matches_points(self) -> Self:
        if self.point_count != len(self.points):
            raise ValueError(
                "point_count must equal the number of returned lattice points"
            )
        return self


class CountLatticePointsResult(StrictModel):
    """The number of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1)
    point_count: int = Field(ge=0)
    representation: str


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
    "MAX_HALFSPACES",
    "MAX_LATTICE_POINTS",
    "MAX_VERTICES",
    "CountLatticePointsResult",
    "EnumerateLatticePointsRequest",
    "EnumerateLatticePointsResult",
    "Halfspace",
    "LatticePoint",
    "LatticePolytopeRequest",
    "Vertex",
]
