"""Request/result models for the bounded rational polytope domain."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_DIMENSION = 6
"""Absolute upper bound on the ambient dimension of a polytope."""

MAX_VERTICES = 64
"""Absolute upper bound on the number of vertices in a V-representation.

The exact convex-hull enumeration is ``O(C(n, d))``; this vertex bound
together with the dimension bound keeps the bounded exact computation
feasible. Polytopes whose hull enumeration exceeds the work bound are
rejected as budget exhaustion.
"""

MAX_FACETS = 64
"""Absolute upper bound on the number of half-spaces in an H-representation."""

COORDINATE_DIGITS = 32_768
"""Per-component digit bound forwarded to the canonical rational validator."""


class Vertex(StrictModel):
    """One rational vertex of a V-representation."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=2, max_length=MAX_DIMENSION
    )


class Halfspace(StrictModel):
    """One rational half-space ``<a, x> <= b`` of an H-representation."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=2, max_length=MAX_DIMENSION
    )
    offset: CanonicalRational


class PolytopeVolumeRequest(StrictModel):
    """A bounded rational polytope in exactly one of the two representations."""

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
        ge=2,
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
            assert self.vertices is not None  # for type checkers
            _validate_vertices(self.vertices, self.dimension_bound)
        else:
            assert self.halfspaces is not None  # for type checkers
            _validate_halfspaces(self.halfspaces, self.dimension_bound)
        return self


def _validate_vertices(vertices: tuple[Vertex, ...], dimension_bound: int) -> None:
    """Validate a V-representation: count, per-component, and dimension bounds."""
    if len(vertices) < 1:
        raise ValueError("`vertices` must be non-empty")
    if len(vertices) > MAX_VERTICES:
        raise ValueError(f"`vertices` exceeds the {MAX_VERTICES}-vertex bound")
    for vertex in vertices:
        for coord in vertex.coordinates:
            require_bounded_rational(
                coord, max_digits=COORDINATE_DIGITS, label="vertex coordinate"
            )
    dim = len(vertices[0].coordinates)
    if dim > dimension_bound:
        raise ValueError(
            f"dimension {dim} exceeds the dimension bound {dimension_bound}"
        )
    for vertex in vertices:
        if len(vertex.coordinates) != dim:
            raise ValueError("all vertices must share one dimension")


def _validate_halfspaces(  # noqa: C901
    halfspaces: tuple[Halfspace, ...], dimension_bound: int
) -> None:
    """Validate an H-representation: count, per-component, and dimension bounds."""
    if len(halfspaces) < 1:
        raise ValueError("`halfspaces` must be non-empty")
    if len(halfspaces) > MAX_FACETS:
        raise ValueError(f"`halfspaces` exceeds the {MAX_FACETS}-facet bound")
    for halfspace in halfspaces:
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
    dim = len(halfspaces[0].coefficients)
    if dim > dimension_bound:
        raise ValueError(
            f"dimension {dim} exceeds the dimension bound {dimension_bound}"
        )
    for halfspace in halfspaces:
        if len(halfspace.coefficients) != dim:
            raise ValueError("all half-spaces must share one dimension")
    for halfspace in halfspaces:
        if all(c.as_fraction() == 0 for c in halfspace.coefficients):
            raise ValueError("half-space coefficients must not all be zero")
    # Bounded-ness and non-emptiness must be decided before any exact
    # enumeration. These checks are the operation's domain filter: an
    # unbounded or empty H-polytope is not a valid request, so it is
    # rejected here as ``ValidationError`` rather than as a host exception
    # after acceptance.
    from jacobian.math.polytope._operations import (
        _is_bounded_h,
        _vertices_from_h_representation,
    )

    if not _is_bounded_h(halfspaces):
        raise ValueError(
            "the H-representation is unbounded; polytope volume requires a bounded polytope"
        )
    verts, _ = _vertices_from_h_representation(halfspaces)
    if not verts:
        raise ValueError("the H-representation defines an empty polytope")


class PolytopeVolumeResult(StrictModel):
    """The exact rational volume of a bounded rational polytope."""

    volume: str
    """The exact rational volume as a reduced ``num/den`` canonical string."""
    dimension: int
    """The ambient dimension of the polytope."""
    representation: str
    """``"vertices"`` or ``"halfspaces"``: the input representation used."""
    evidence: str = "COMPUTED"
    """The volume is exact rational, not a floating-point approximation."""


__all__ = [
    "MAX_DIMENSION",
    "MAX_FACETS",
    "MAX_VERTICES",
    "Halfspace",
    "PolytopeVolumeRequest",
    "PolytopeVolumeResult",
    "Vertex",
]
