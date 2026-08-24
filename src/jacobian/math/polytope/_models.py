"""Request/result models for the bounded rational polytope domain."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json

MAX_DIMENSION = 6
"""Absolute upper bound on the ambient dimension of a polytope."""

MAX_FACET_DIMENSION = 7
"""Ambient-dimension bound for complete V-representation facet profiles.

This is deliberately one dimension wider than exact volume: the 14-vertex
0/1 counterexample motivating the facet operation is seven-dimensional.
``Vertex`` is the shared V-representation value, while individual operations
still publish and enforce their own dimensional envelopes.
"""

MAX_FACET_COORDINATE_DIGITS = 32
"""Per-component input-height bound for exact facet enumeration.

Candidate supporting hyperplanes are determinants of rational coordinate
differences. This conservative height keeps those private intermediates and
the primitive integer facet rows comfortably inside the canonical transport
limit for the complete returned profile.
"""

MAX_FACET_SIGN_TESTS = 5_000_000
"""Maximum candidate-hyperplane/vertex side tests in one enumeration pass."""

MAX_FACET_TOTAL_SIGN_TESTS = 3 * MAX_FACET_SIGN_TESTS
"""Maximum candidate-side tests for request admission, execution, and exact
result replay together."""

MAX_COMPUTED_FACETS = 256
"""Maximum number of canonical facets materialized by one result."""

MAX_FACET_INCIDENCES = 16_384
"""Maximum total source-row/facet incidences materialized by one result."""

MAX_VERTICES = 64
"""Absolute upper bound on the number of vertices in a V-representation.

The exact convex-hull enumeration is ``O(C(n, d))``; this vertex bound
together with the dimension bound keeps the bounded exact computation
feasible. Polytopes whose hull enumeration exceeds the work bound are
rejected as budget exhaustion.
"""

MAX_FACETS = 64
"""Absolute upper bound on the number of half-spaces in an H-representation."""

MAX_HULL_SUBFACETS = 200_000
"""Ceiling on the d-subsets the exact hull enumeration may consider.

The hull work couples vertex count with ambient dimension: after
duplicate points are removed, enumerating the hull of ``n`` distinct
vertices in dimension ``d`` considers ``C(n, d)`` d-subsets, so requests
with ``C(n, d) > 200_000`` are rejected as budget exhaustion. The bound
lives here so the published request-schema descriptions quote the same
constant the admission check enforces.
"""

MAX_BOUNDEDNESS_COMBINATIONS = 700_000
"""Ceiling on the row combinations the H-representation boundedness
precheck may consider.

Deciding boundedness exactly enumerates the facets of the convex hull of
the row normals, which considers ``C(m, d)`` d-subsets of the ``m``
distinct half-spaces in ambient dimension ``d``. Duplicate rows
(identical up to a common positive factor) are removed first, so
redundant copies neither change the decision nor inflate the estimate;
requests with ``C(m, d) > 700_000`` distinct rows are rejected as budget
exhaustion. The bound lives here so the published request-schema
description quotes the same constant the admission check enforces.
"""

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

    if isinstance(value, Fraction):
        return value.numerator, value.denominator
    p = getattr(value, "p", None)
    q = getattr(value, "q", None)
    if p is None or q is None:
        raise TypeError("value is not a rational")
    return int(p), int(q)


def _point_digit_lengths(point: Sequence[object]) -> list[tuple[int, int]]:
    """Decimal digit lengths of each coordinate's reduced components."""

    from jacobian.canonical import format_canonical_integer

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


def _require_interval_volume_within_result_bound(
    points: Sequence[Sequence[object]],
) -> None:
    """Bound the one-dimensional volume ``max - min`` of the given points.

    The reduced difference has a denominator dividing the product of the
    two endpoint denominators and a numerator bounded by the cross-term
    ``|p_a q_b - p_b q_a|``, so admission measures decimal component
    lengths: the largest numerator length plus the largest denominator
    length, and the sum of the two largest denominator lengths.
    """

    from jacobian.canonical import format_canonical_integer

    values = [_rational_pq(point[0]) for point in points]
    numerator_digits = (
        max(len(format_canonical_integer(abs(p))) for p, _ in values)
        + max(len(format_canonical_integer(q)) for _, q in values)
        + 2
    )
    den_lengths = sorted(
        (len(format_canonical_integer(q)) for _, q in values),
        reverse=True,
    )
    top_two = sum(den_lengths[:2])
    if (
        numerator_digits > MAX_RESULT_COMPONENT_DIGITS
        or top_two + 2 > MAX_RESULT_COMPONENT_DIGITS
    ):
        raise ValueError(
            "coordinate magnitudes can grow the exact volume beyond the "
            f"{MAX_RESULT_COMPONENT_DIGITS}-digit canonical rational "
            "result bound"
        )


def _require_triangulated_volume_within_result_bound(
    table: list[list[tuple[int, int]]],
    triangulation: list[tuple[int, ...]],
    dim: int,
) -> None:
    """Bound the summed simplex volumes against the canonical component limit.

    With ``R_v`` the total denominator-digit count of vertex ``v``'s
    coordinates, one simplex contributes a common denominator of at most
    ``sum(R_v)`` digits and its scaled Hadamard determinant numerator at
    most ``sum(max_k(n_vk + R_v) + 1)`` digits.  Each simplex's numerator
    estimate dominates its denominator estimate, so the sum over all
    simplices bounds both components of the combined fraction; summation
    carries add a small slack.  The bound is conservative: it may reject
    inputs whose concrete volume happens to be short, never accepts one
    that cannot be represented.
    """

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
        denominator_total += sum(q for idx in simplex for _, q in table[idx])
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


def _deduplicate_exact_points(
    points: Sequence[Sequence[object]],
) -> list[Sequence[object]]:
    """Drop repeated points, preserving first-seen order."""

    seen: set[tuple[tuple[int, int], ...]] = set()
    unique: list[Sequence[object]] = []
    for point in points:
        key = tuple(_rational_pq(coord) for coord in point)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def require_volume_components_within_result_bound(
    points: Sequence[Sequence[object]],
    dim: int,
) -> None:
    """Reject inputs whose exact summed volume cannot fit the canonical type.

    The kernel sums simplex determinants over a whole triangulation, so
    admission must account for denominators contributed by *all* simplices,
    not only ``dim + 1`` vertices.  The guard mirrors the execution
    pipeline — exact deduplication, redundant-vertex filtering,
    triangulation — so an empty or failed triangulation here means the
    kernel returns exact volume zero, which is always representable.  The
    combinatorial hull-work bound applies to the *unique* points before
    any enumeration runs, exactly as the kernel counts its own work after
    deduplication, so repeated points neither inflate the budget nor let
    unrepresentable inputs slip past; every caller of this guard (request
    validation or the native wrapper) is protected from unguarded hull
    work.
    """

    if dim == 1:
        # Mirror the kernel's one-dimensional pipeline: deduplicate
        # exactly, and a hull with fewer than two distinct coordinates is
        # degenerate with exact volume zero, which is always representable.
        unique = _deduplicate_exact_points(points)
        if len(unique) < 2:
            return
        _require_interval_volume_within_result_bound(unique)
        return

    from jacobian.math.polytope._operations import (
        _filter_redundant_vertices,
        _triangulate,
    )

    # Deduplicate exactly as the kernel does before any admission work:
    # the budget below measures the hull enumeration the kernel actually
    # performs on unique points, and duplicate points would otherwise
    # break the polygonal adjacency into an empty triangulation and let
    # unrepresentable inputs skip admission entirely.
    pts = [list(point) for point in _deduplicate_exact_points(points)]
    try:
        subfacets = math.comb(len(pts), dim)
    except ValueError:
        subfacets = 10**18
    if subfacets > MAX_HULL_SUBFACETS:
        raise ValueError(
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacets} > {MAX_HULL_SUBFACETS} d-subsets)"
        )
    pts = _filter_redundant_vertices(pts, dim)
    if len(pts) < dim + 1:
        return
    triangulation = _triangulate(pts, dim)
    if not triangulation:
        return
    table = [_point_digit_lengths(row) for row in pts]
    _require_triangulated_volume_within_result_bound(table, triangulation, dim)


class Vertex(StrictModel):
    """One rational vertex of a V-representation."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_FACET_DIMENSION
    )


class PrimitiveFacet(StrictModel):
    """One canonically scaled supporting inequality and its source incidences.

    ``halfspace`` carries the supporting inequality ``<a, x> <= b`` in the
    domain's shared H-representation value. A computed facet composes
    unchanged into any H-representation consumer whose admitted ambient
    dimension covers this profile's: ``polytope.volume.compute`` caps
    dimension at ``MAX_DIMENSION = 6``, while profiles here may reach
    ``MAX_FACET_DIMENSION = 7``. Its entries are integers whose only common
    divisor is one, and its orientation is the unique one satisfied by every
    source vertex. ``source_vertex_indices`` is the sorted complete set of
    positions in the ordered source V-representation lying on the supporting
    hyperplane; repeated source rows remain distinct positions.
    """

    halfspace: Halfspace = Field(
        description=(
            "Supporting inequality <a, x> <= b with primitive integer entries, "
            "oriented so every source vertex satisfies it."
        ),
    )
    source_vertex_indices: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "Strictly increasing positions of all source V-representation rows "
            "on this facet."
        ),
    )

    @model_validator(mode="after")
    def require_primitive_normal_and_indices(self) -> Self:
        if all(
            coefficient.as_fraction() == 0
            for coefficient in self.halfspace.coefficients
        ):
            raise ValueError("facet inequality must have a nonzero normal")
        entries = [
            Fraction(*value.as_integer_ratio())
            for value in (*self.halfspace.coefficients, self.halfspace.offset)
        ]
        if any(entry.denominator != 1 for entry in entries):
            raise ValueError("facet inequality entries must be integers")
        gcd = 0
        for entry in entries:
            gcd = math.gcd(gcd, abs(int(entry)))
        if gcd != 1:
            raise ValueError("facet inequality must be primitive over the integers")
        if any(
            right <= left
            for left, right in zip(
                self.source_vertex_indices,
                self.source_vertex_indices[1:],
                strict=False,
            )
        ):
            raise ValueError("facet source vertex indices must be strictly increasing")
        return self


class FacetIncidenceResult(StrictModel):
    """Complete source-bound facet profile of a full-dimensional rational polytope."""

    vertices: tuple[Vertex, ...] = Field(
        min_length=2,
        max_length=MAX_VERTICES,
        description=(
            "The exact ordered V-representation from which the profile was computed; "
            "facet incidences index this tuple."
        ),
    )
    dimension: int = Field(ge=1, le=MAX_FACET_DIMENSION)
    facets: tuple[PrimitiveFacet, ...] = Field(
        min_length=2,
        max_length=MAX_COMPUTED_FACETS,
        description=(
            "All maximal codimension-one faces, sorted lexicographically by their "
            "canonical primitive supporting inequalities."
        ),
    )

    @model_validator(mode="after")
    def require_complete_source_bound_profile(self) -> Self:
        if any(len(vertex.coordinates) != self.dimension for vertex in self.vertices):
            raise ValueError(
                "every source vertex must have exactly `dimension` coordinates"
            )
        for vertex in self.vertices:
            for coordinate in vertex.coordinates:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_FACET_COORDINATE_DIGITS,
                    label="facet-profile vertex coordinate",
                )
        if (
            sum(len(facet.source_vertex_indices) for facet in self.facets)
            > MAX_FACET_INCIDENCES
        ):
            raise ValueError(
                "facet profile exceeds the "
                f"{MAX_FACET_INCIDENCES}-incidence result bound"
            )
        from jacobian.math.polytope._operations import _computed_facets_from_vertices

        expected = _computed_facets_from_vertices(self.vertices, self.dimension)
        if self.facets != expected:
            raise ValueError(
                "facets must be the complete canonical source-bound facet profile"
            )
        try:
            encode_strict_json(self.model_dump(mode="json"), limits=CanonicalLimits())
        except ValueError as exc:
            raise ValueError(
                "facet profile exceeds the canonical JSON output bound"
            ) from exc
        return self


class FacetIncidenceRequest(StrictModel):
    """A full-dimensional bounded rational V-representation for facet enumeration."""

    vertices: tuple[Vertex, ...] = Field(
        min_length=2,
        max_length=MAX_VERTICES,
        description=(
            "Ordered rational V-representation. The points must affinely span their "
            "ambient dimension; lower-dimensional hulls are rejected because this "
            "operation returns ambient codimension-one facets. Repeated source rows "
            "are retained for incidence binding but create no candidate hyperplanes "
            "or candidate side tests, so admission requires m*C(m,d) <= "
            f"{MAX_FACET_SIGN_TESTS} candidate-side tests, where m is the number of "
            "distinct rows and every candidate hyperplane spanned by those distinct "
            "rows is side-tested against exactly those distinct rows; the final-facet "
            "incidence scans then range over all n source positions and are bounded "
            "by the materialized-profile result limits below. Both charges apply "
            "in each of request admission, execution, and exact result replay "
            f"({MAX_FACET_TOTAL_SIGN_TESTS} total). Admission materializes the "
            "complete bounded enumeration before accepting a request, so its exact "
            f"facet and incidence counts are proven to fit the "
            f"{MAX_COMPUTED_FACETS}-facet and "
            f"{MAX_FACET_INCIDENCES}-incidence result limits."
        ),
    )
    dimension_bound: int = Field(
        default=MAX_FACET_DIMENSION,
        ge=1,
        le=MAX_FACET_DIMENSION,
        description="Maximum admitted ambient dimension for this facet profile.",
    )

    @model_validator(mode="after")
    def require_admissible_full_dimensional_profile(self) -> Self:
        dimension = len(self.vertices[0].coordinates)
        if dimension > self.dimension_bound:
            raise ValueError(
                f"dimension {dimension} exceeds the dimension bound {self.dimension_bound}"
            )
        if any(len(vertex.coordinates) != dimension for vertex in self.vertices):
            raise ValueError("all vertices must share one dimension")
        for vertex in self.vertices:
            for coordinate in vertex.coordinates:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_FACET_COORDINATE_DIGITS,
                    label="facet-profile vertex coordinate",
                )
        from jacobian.math.polytope._operations import _computed_facets_from_vertices

        # Materializing the complete bounded enumeration here proves the
        # facet and incidence result bounds during request validation, so an
        # admitted request cannot discover an oversized profile only in the
        # execution backend.
        _computed_facets_from_vertices(self.vertices, dimension)
        return self


class Halfspace(StrictModel):
    """One rational half-space ``<a, x> <= b`` of an H-representation.

    The normal ``a`` must be nonzero: at least one coefficient entry must
    differ from zero.  A row with all-zero coefficients is a tautology or
    contradiction, not a half-space, and is rejected as a typed request
    error rather than silently changing the polytope.
    """

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_FACET_DIMENSION,
        description=(
            "Normal vector a of the half-space <a, x> <= b; at least one "
            "entry must be nonzero (all-zero rows are rejected)."
        ),
    )
    offset: CanonicalRational


class PolytopeVolumeRequest(StrictModel):
    """A bounded rational polytope in exactly one of the two representations.

    Admission enforces a work bound that couples vertex count with ambient
    dimension: after duplicate points are removed, the exact hull
    enumeration considers ``C(n, d)`` d-subsets of ``n`` distinct vertices
    in dimension ``d``, and requests with ``C(n, d) > 200000`` are rejected.
    The same limit applies to the vertex set derived from an
    H-representation, whose own rows are additionally bounded by
    ``C(m, d) <= 700000`` on the distinct half-spaces (see the field
    descriptions for the exact published rules).
    """

    vertices: tuple[Vertex, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "V-representation: the vertices of the convex hull. "
            "Mutually exclusive with ``halfspaces``. "
            f"Coupled hull-work bound: after duplicate points are removed, "
            f"admission requires C(n, d) <= {MAX_HULL_SUBFACETS} on the n "
            "distinct vertices in ambient dimension d (the exact hull "
            "enumeration considers every d-subset); larger requests are "
            "rejected. Within the 64-vertex maximum this admits up to 64 "
            "distinct vertices for d <= 3, 48 for d = 4, 31 for d = 5, and "
            "25 for d = 6."
        ),
    )
    halfspaces: tuple[Halfspace, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_FACETS,
        description=(
            "H-representation: the half-spaces ``<a_i, x> <= b_i``. "
            "Mutually exclusive with ``vertices``. Each half-space must "
            "have a nonzero normal: a row whose coefficients are all zero "
            "is rejected. Coupled hull-work bound: duplicate rows "
            "(identical up to a common positive factor) are removed, then "
            f"admission requires C(m, d) <= {MAX_BOUNDEDNESS_COMBINATIONS} "
            "on the m distinct half-spaces in ambient dimension d (the "
            "boundedness precheck exactly enumerates the hull of the row "
            "normals); within the 64-row maximum this admits 64 distinct "
            "half-spaces for d <= 4, 40 for d = 5, and 30 for d = 6. The "
            f"derived vertex set is then subject to the C(n, d) <= "
            f"{MAX_HULL_SUBFACETS} hull-work bound published on the "
            "vertices field."
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
    from jacobian.math.polytope._operations import _vertices_from_v_representation

    # Exact-volume growth is bounded over the whole triangulation, so the
    # same admission runs on the rational points themselves; it applies
    # the combinatorial hull-work bound after exact deduplication,
    # mirroring the kernel pipeline.
    points, resolved_dim = _vertices_from_v_representation(vertices)
    require_volume_components_within_result_bound(points, resolved_dim)


def _require_admissible_h_vertices(halfspaces: tuple[Halfspace, ...], dim: int) -> None:
    """Admit the derived vertex set of an H-representation.

    Bounded-ness and non-emptiness must be decided before any exact
    enumeration: an unbounded or empty H-polytope is not a valid request,
    so it is rejected here as ``ValidationError`` rather than as a host
    exception after acceptance.  The derived vertices then drive the same
    brute-force hull enumeration and exact-volume growth bound as a
    caller-supplied V-representation, so the identical combinatorial and
    result-size admission applies before accepting the request.
    """

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
    subfacets = math.comb(len(verts), dim)
    if subfacets > MAX_HULL_SUBFACETS:
        raise ValueError(
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacets} > {MAX_HULL_SUBFACETS} d-subsets)"
        )
    # Solved vertices can carry more digits than the declaring half-space
    # coefficients, so measure them directly.
    require_volume_components_within_result_bound(verts, dim)


def _validate_halfspaces(
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
    _require_admissible_h_vertices(halfspaces, dim)


class PolytopeVolumeResult(StrictModel):
    """The exact rational volume of a bounded rational polytope."""

    volume: CanonicalRational
    """The exact rational volume as a canonical reduced rational."""
    dimension: int
    """The ambient dimension of the polytope."""
    representation: str
    """``"vertices"`` or ``"halfspaces"``: the input representation used."""


__all__ = [
    "MAX_BOUNDEDNESS_COMBINATIONS",
    "MAX_COMPUTED_FACETS",
    "MAX_DIMENSION",
    "MAX_FACETS",
    "MAX_FACET_COORDINATE_DIGITS",
    "MAX_FACET_DIMENSION",
    "MAX_FACET_INCIDENCES",
    "MAX_FACET_SIGN_TESTS",
    "MAX_FACET_TOTAL_SIGN_TESTS",
    "MAX_HULL_SUBFACETS",
    "MAX_VERTICES",
    "FacetIncidenceRequest",
    "FacetIncidenceResult",
    "Halfspace",
    "PolytopeVolumeRequest",
    "PolytopeVolumeResult",
    "PrimitiveFacet",
    "Vertex",
]
