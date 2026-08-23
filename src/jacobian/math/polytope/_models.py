"""Request/result models for the bounded rational polytope domain."""

from __future__ import annotations

from collections.abc import Sequence
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

MAX_RESULT_COMPONENT_DIGITS = 32_768
"""Digit bound each exact-volume component must respect to be returnable.

The volume is a canonical rational whose components cannot exceed the
global ``CanonicalRational`` limit; requests whose exact volume can
provably leave that domain are rejected at admission.
"""


def _rational_pq(value: object) -> tuple[int, int]:
    """Return ``(numerator, denominator)`` from a Fraction or SymPy Rational."""

    p = getattr(value, "p", None)
    if p is not None:
        return int(p), int(value.q)
    return int(value.numerator), int(value.denominator)


def require_volume_components_within_result_bound(
    points: Sequence[Sequence[object]],
    dim: int,
) -> None:
    """Reject inputs whose exact summed volume cannot fit the canonical type.

    The kernel sums simplex determinants over a whole triangulation, so
    admission must account for denominators contributed by *all* simplices,
    not only ``dim + 1`` vertices.  With ``R_v`` the total denominator-digit
    count of vertex ``v``'s coordinates, one simplex contributes a common
    denominator of at most ``sum(R_v)`` digits, its scaled Hadamard
    determinant numerator at most ``sum(max_k(n_vk + R_v) + 1)`` digits, and
    the sum over ``T`` simplices multiplies these by ``T`` (common
    denominator product) plus ``dim!`` and summation carries.  The bound is
    conservative: it may reject inputs whose concrete volume happens to be
    short, never accepts one that cannot be represented.
    """

    from jacobian.canonical import format_canonical_integer

    def digits(point: Sequence[object]) -> list[tuple[int, int]]:
        row: list[tuple[int, int]] = []
        for coord in point:
            p, q = _rational_pq(coord)
            row.append(
                (
                    len(format_canonical_integer(abs(p))),
                    len(format_canonical_integer(q)),
                )
            )
        return row

    if dim == 1:
        values = [_rational_pq(point[0]) for point in points]
        num_digits = max(n for n, _ in values) + max(q for _, q in values) + 2
        den_digits = max(q for _, q in values) + 2
        if (
            num_digits > MAX_RESULT_COMPONENT_DIGITS
            or den_digits > MAX_RESULT_COMPONENT_DIGITS
        ):
            raise ValueError(
                "coordinate magnitudes can grow the exact volume beyond the "
                f"{MAX_RESULT_COMPONENT_DIGITS}-digit canonical rational "
                "result bound"
            )
        return

    from jacobian.math.polytope._operations import (
        _filter_redundant_vertices,
        _triangulate,
    )

    pts = [list(point) for point in points]
    pts = _filter_redundant_vertices(pts, dim)
    if len(pts) < dim + 1:
        return
    triangulation = _triangulate(pts, dim)
    if not triangulation:
        return
    table = [digits(row) for row in pts]
    numerator_total = 0
    denominator_total = 0
    for simplex in triangulation:
        det_digits = 0
        for idx in simplex:
            row = table[idx]
            row_den = sum(q for _, q in row)
            row_max = max(n + row_den for n, _ in row)
            det_digits += row_max + 2
        numerator_total += det_digits
        denominator_total += sum(
            q for idx in simplex for _, q in table[idx]
        )
    carry = dim + len(str(len(triangulation))) + 4
    if (
        numerator_total + carry > MAX_RESULT_COMPONENT_DIGITS
        or denominator_total + carry > MAX_RESULT_COMPONENT_DIGITS
    ):
        raise ValueError(
            "coordinate magnitudes can grow the exact volume beyond the "
            f"{MAX_RESULT_COMPONENT_DIGITS}-digit canonical rational "
            "result bound"
        )


def _require_volume_within_result_bound(
    component_numerator_digits: int,
    component_denominator_digits: int,
    dim: int,
) -> None:
    """Reject inputs whose exact volume cannot fit the canonical result type.

    Each triangulation simplex uses ``dim + 1`` vertices. Scaling every
    vertex by its component denominators bounds the scaled determinant
    numerator by Hadamard at ``sum(n_i + q_i)`` digits, while the common
    denominator product contributes ``sum(q_i) + log10(dim!)`` digits;
    the sum over at most ``MAX_VERTICES`` simplices adds two further
    digits. The bound is conservative: it may reject inputs whose
    concrete volume happens to be short, never accepts one that cannot
    be represented.
    """

    bound = (dim + 1) * (
        component_numerator_digits + 2 * component_denominator_digits + 1
    ) + dim + 4
    if bound > MAX_RESULT_COMPONENT_DIGITS:
        raise ValueError(
            "coordinate magnitudes can grow the exact volume beyond the "
            f"{MAX_RESULT_COMPONENT_DIGITS}-digit canonical rational "
            "result bound"
        )


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
    offset: CanonicalRational


class PolytopeVolumeRequest(StrictModel):
    """A bounded rational polytope in exactly one of the two representations."""

    vertices: tuple[Vertex, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "V-representation: the vertices of the convex hull. "
            "Mutually exclusive with ``halfspaces``."
        ),
    )
    halfspaces: tuple[Halfspace, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FACETS,
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
    numerator_digits = 0
    denominator_digits = 0
    for vertex in vertices:
        for coord in vertex.coordinates:
            require_bounded_rational(
                coord, max_digits=COORDINATE_DIGITS, label="vertex coordinate"
            )
            numerator_digits = max(numerator_digits, len(coord.num.lstrip("-")))
            denominator_digits = max(denominator_digits, len(coord.den))
    dim = len(vertices[0].coordinates)
    if dim > dimension_bound:
        raise ValueError(
            f"dimension {dim} exceeds the dimension bound {dimension_bound}"
        )
    for vertex in vertices:
        if len(vertex.coordinates) != dim:
            raise ValueError("all vertices must share one dimension")
    # Combinatorial admission first: C(n, d) d-subsets for hull
    # enumeration.  The operation's brute-force hull needs to consider each
    # d-subset; reject here so neither execution nor the triangulation-aware
    # growth bound below ever enumerates beyond this budget.  This admits
    # the 5-cube (C(32,5)=201376) test threshold but rejects larger hulls.
    import math

    from jacobian.math.polytope._operations import (
        MAX_HULL_SUBFACETS,
        _vertices_from_v_representation,
    )

    try:
        subfacets = math.comb(len(vertices), dim)
    except ValueError:
        subfacets = 10**18
    if subfacets > MAX_HULL_SUBFACETS:
        raise ValueError(
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacets} > {MAX_HULL_SUBFACETS} d-subsets)"
        )
    # Exact-volume growth is bounded over the whole triangulation, so the
    # same admission runs on the rational points themselves.
    points, resolved_dim = _vertices_from_v_representation(vertices)
    require_volume_components_within_result_bound(points, resolved_dim)


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
    # The derived vertex set drives the same brute-force hull enumeration as
    # a caller-supplied V-representation, so the identical combinatorial
    # admission applies before accepting the request.  Otherwise execution
    # would raise the hull-budget error as a host exception after acceptance.
    import math

    from jacobian.math.polytope._operations import MAX_HULL_SUBFACETS

    subfacets = math.comb(len(verts), dim)
    if subfacets > MAX_HULL_SUBFACETS:
        raise ValueError(
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacets} > {MAX_HULL_SUBFACETS} d-subsets)"
        )
    # Derived vertices also drive the exact-volume growth bound over the
    # whole triangulation; solved vertices can carry more digits than the
    # declaring half-space coefficients, so measure them directly.
    require_volume_components_within_result_bound(verts, dim)


class PolytopeVolumeResult(StrictModel):
    """The exact rational volume of a bounded rational polytope."""

    volume: CanonicalRational
    """The exact rational volume as a canonical reduced rational."""
    dimension: int
    """The ambient dimension of the polytope."""
    representation: str
    """``"vertices"`` or ``"halfspaces"``: the input representation used."""


__all__ = [
    "MAX_DIMENSION",
    "MAX_FACETS",
    "MAX_VERTICES",
    "Halfspace",
    "PolytopeVolumeRequest",
    "PolytopeVolumeResult",
    "Vertex",
]
