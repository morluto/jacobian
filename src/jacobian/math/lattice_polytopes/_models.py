"""Typed wire contracts for bounded rational lattice-polytope operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
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

Both operations fail closed with a budget error before enumerating or
counting more lattice points than this. ``enumerate`` lists every point
up to the bound; ``count`` returns the count without materializing the
full list but still refuses to count beyond the bound.
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
            "V-representation: the vertices of the convex hull. "
            "Mutually exclusive with ``halfspaces``."
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

    def dimension(self) -> int:
        """Return the ambient dimension implied by the chosen representation."""
        if self.vertices is not None:
            return len(self.vertices[0].coordinates)
        assert self.halfspaces is not None
        return len(self.halfspaces[0].coefficients)


class LatticePoint(StrictModel):
    """One lattice point, as a tuple of canonical integers."""

    coordinates: tuple[str, ...] = Field(min_length=1)


class EnumerateLatticePointsResult(StrictModel):
    """The complete list of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1)
    point_count: int = Field(ge=0)
    points: tuple[LatticePoint, ...]
    representation: str


class CountLatticePointsResult(StrictModel):
    """The number of lattice points inside a bounded rational polytope."""

    dimension: int = Field(ge=1)
    point_count: int = Field(ge=0)
    representation: str


__all__ = [
    "MAX_BOUND_SPAN",
    "MAX_DIMENSION",
    "MAX_HALFSPACES",
    "MAX_LATTICE_POINTS",
    "MAX_VERTICES",
    "CountLatticePointsResult",
    "EnumerateLatticePointsResult",
    "Halfspace",
    "LatticePoint",
    "LatticePolytopeRequest",
    "Vertex",
]
