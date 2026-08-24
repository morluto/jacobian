"""Request/result models for the bounded rational polytope domain."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Annotated, Any, Self

from pydantic import (
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

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

MAX_SUPPORT_COMPONENT_DIGITS = 150
"""Per-component digit cap for rational polytope support inputs.

The support value alone would permit a much larger cap, but canonical
V-polytope validation also executes the existing exact hull-facet kernel. In
dimension six, a facet normal has at most ``36D + 3`` digits after row-wise
denominator clearing, and its active-normal rank proof has minors of at most
``6 * (36D + 3) + 3 = 216D + 21`` digits. With ``D = 150`` every exact
intermediate stays below the 32,768-digit canonical rational limit (the
largest bound is 32,421 digits). The support dot product is smaller at
``2*d*D + 2`` digits. This one cap therefore bounds the admitted support
inputs and the exact construction/replay work required by the public
V-polytope value.
"""

MAX_SUPPORT_VERTEX_SUBSETS = 100_000
"""Maximum ``C(n, d)`` exact subfacets used to verify a V-polytope.

The support kernel itself is a linear pass. Its canonical V-polytope value
also proves that every declared generator is an extreme vertex, using the
existing exact hull-facet kernel. This bound applies before that proof
materializes its candidate subsets; its distinct orientation-test bound is
declared separately.
"""

MAX_SUPPORT_ORIENTATION_TESTS = 500_000
"""Maximum exact orientation determinants in the V-polytope hull proof.

For every candidate ``d``-subset, the hull kernel tests each of the ``n-d``
remaining vertices against its supporting hyperplane. Admission charges the
complete product ``C(n,d) * (n-d)`` before materializing either family.
"""

CoordinateAxis = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, strict=True),
]
"""One coordinate identifier in an ordered labelled rational space."""


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


def _raw_field_value(owner: object, name: str) -> object:
    """Read one named field from a raw dict or an already-built model."""

    if isinstance(owner, dict):
        return owner.get(name)
    return getattr(owner, name, None)


def _iter_raw_entries(owner: object, name: str) -> Iterator[object]:
    """Yield one raw payload collection's entries when it is a sequence."""

    entries = _raw_field_value(owner, name)
    if isinstance(entries, (list, tuple)):
        yield from entries


def _preflight_raw_support_components(data: object) -> object:
    """Measure every authored support component against the envelope.

    Shared by the support request and result models so a payload whose
    components sit between the operation's 150-digit envelope and the
    global canonical limit is rejected before nested V-polytope parsing
    constructs (and canonically proves) the retained source.
    """

    if not isinstance(data, dict):
        return data
    canonical: Any = _tuple_canonical_containers(data)
    _require_raw_support_covector_admissible(canonical)
    for vertex in _iter_raw_entries(canonical.get("polytope"), "vertices"):
        for component in _iter_raw_entries(vertex, "coordinates"):
            _require_raw_component_within_support_envelope(
                component,
                "polytope vertex coordinate",
            )
    for component in _iter_raw_entries(canonical.get("covector"), "components"):
        _require_raw_component_within_support_envelope(
            component,
            "covector component",
        )
    return canonical


def _require_raw_component_within_support_envelope(
    component: object,
    label: str,
) -> None:
    """Measure one authored rational payload against the support envelope.

    The reduced numerator/denominator strings are read exactly as
    ``require_bounded_rational`` measures them, but without constructing
    any model; unrecognized shapes fall through to ordinary nested
    validation errors.
    """

    if isinstance(component, CanonicalRational):
        num, den = component.num, component.den
    elif isinstance(component, dict):
        raw_num = component.get("num")
        raw_den = component.get("den")
        if not isinstance(raw_num, str) or not isinstance(raw_den, str):
            return
        num, den = raw_num, raw_den
    else:
        return
    if max(len(num.lstrip("-")), len(den.lstrip("-"))) > (MAX_SUPPORT_COMPONENT_DIGITS):
        raise ValueError(
            f"{label} exceeds the {MAX_SUPPORT_COMPONENT_DIGITS}-digit bound"
        )


def _raw_space_axes(space: object) -> tuple[object, ...] | None:
    """Read one raw or built coordinate space's declared axes.

    Unrecognized payload shapes return ``None`` so ordinary canonical
    validation reports them with the published schema errors.
    """

    if isinstance(space, RationalCoordinateSpace):
        return tuple(space.axes)
    if isinstance(space, dict):
        axes = space.get("axes")
        if isinstance(axes, (list, tuple)):
            return tuple(axes)
    return None


def _require_raw_canonical_rational_component(
    component: object,
    label: str,
) -> None:
    """Reject component shapes that cannot construct ``CanonicalRational``.

    A canonical rational parses only from its serialized ``num``/``den``
    object or an already-built value — the component fields are strict
    canonical strings and the model forbids extra keys — so any other
    authored shape is certain to be rejected by nested validation, only
    after the polytope field has paid its exact hull proof.
    """

    if isinstance(component, CanonicalRational):
        return
    if (
        isinstance(component, dict)
        and set(component) == {"num", "den"}
        and isinstance(component["num"], str)
        and isinstance(component["den"], str)
    ):
        return
    raise ValueError(f"{label} must be a canonical rational")


def _require_raw_coordinate_space(value: object, label: str) -> tuple[str, ...]:
    """Mirror ``RationalCoordinateSpace`` on one raw payload value.

    Returns the declared axis labels when the raw space satisfies every
    published constraint (closed ``axes`` field, non-empty sequence of at
    most ``MAX_DIMENSION`` short unique string labels); any violation
    raises here because ordinary nested validation rejects it too, only
    after the hull proof has run.
    """

    if isinstance(value, RationalCoordinateSpace):
        return tuple(value.axes)
    if not isinstance(value, dict) or set(value) != {"axes"}:
        raise ValueError(f"{label} space must be an object with axes")
    axes = value["axes"]
    if not isinstance(axes, (list, tuple)) or not axes:
        raise ValueError(f"{label} space axes must be a non-empty sequence")
    if len(axes) > MAX_DIMENSION:
        raise ValueError(f"{label} space must declare at most {MAX_DIMENSION} axes")
    if any(not isinstance(axis, str) or not 1 <= len(axis) <= 64 for axis in axes):
        raise ValueError(f"{label} space axes must be short string labels")
    if len(set(axes)) != len(axes):
        raise ValueError("coordinate axes must be unique")
    return tuple(axes)


def _require_raw_support_covector_admissible(canonical: Any) -> None:
    """Gate the covector half of one raw support payload.

    Pydantic parses declared fields in order and aggregates nested
    errors, so a raw ``math.run`` payload whose ``polytope`` is valid
    near the hull envelope pays the complete exact extremality proof —
    up to the published orientation-test bound — before a missing
    covector, malformed components, dimension mismatch, or foreign
    space is reported. This gate mirrors only the covector-level
    constraints nested validation rejects anyway: presence, closed
    field set, component container and per-component canonical-rational
    shapes, the declared-axis match, and raw space agreement with the
    polytope.
    """

    covector = canonical.get("covector")
    if covector is None:
        raise ValueError("covector must be provided")
    if isinstance(covector, RationalCovector):
        return
    if not isinstance(covector, dict) or set(covector) != {"space", "components"}:
        raise ValueError("covector must be an object with space and components")
    components = covector["components"]
    if not isinstance(components, (list, tuple)):
        raise ValueError("covector components must be a sequence")
    if not components:
        raise ValueError("covector components must be a non-empty sequence")
    if len(components) > MAX_DIMENSION:
        raise ValueError(
            f"covector components must carry at most {MAX_DIMENSION} entries"
        )
    for component in components:
        _require_raw_canonical_rational_component(component, "covector component")
    axes = _require_raw_coordinate_space(covector["space"], "covector")
    if len(components) != len(axes):
        raise ValueError("covector components must use the declared coordinate axis")
    polytope_axes = _raw_space_axes(
        _raw_field_value(canonical.get("polytope"), "space")
    )
    if (
        polytope_axes is not None
        and all(isinstance(axis, str) for axis in polytope_axes)
        and tuple(polytope_axes) != axes
    ):
        raise ValueError("polytope and covector must use the same coordinate space")


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
        min_length=1, max_length=MAX_DIMENSION
    )


class RationalCoordinateSpace(StrictModel):
    """One ordered labelled rational coordinate space.

    Coordinate order is mathematical data: a covector component can only
    pair with the point coordinate named by the same position in this axis.
    """

    axes: tuple[CoordinateAxis, ...] = Field(min_length=1, max_length=MAX_DIMENSION)

    @model_validator(mode="after")
    def require_distinct_axes(self) -> Self:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("coordinate axes must be unique")
        return self


class RationalPolytopeVertex(StrictModel):
    """A labelled exact vertex in a rational coordinate space."""

    vertex_id: str = Field(min_length=1, max_length=64)
    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_DIMENSION,
    )


class RationalVPolytope(StrictModel):
    """A full-dimensional bounded rational polytope by its exact vertices.

    The vertices are a canonical V-representation: their IDs are strictly
    ordered, their coordinate tuples are distinct, and each one is an exact
    extreme vertex of the hull. Lower-dimensional polytopes need explicit
    intrinsic affine coordinates and are intentionally outside this first
    support-operation contract.
    """

    space: RationalCoordinateSpace
    vertices: tuple[RationalPolytopeVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "Complete irredundant vertex family, ordered strictly by vertex_id. "
            "The exact extremality proof requires C(n,d) <= "
            f"{MAX_SUPPORT_VERTEX_SUBSETS} candidate subfacets and C(n,d) * "
            f"(n-d) <= {MAX_SUPPORT_ORIENTATION_TESTS} orientation tests."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_full_dimensional_vertices(self) -> Self:
        dimension = len(self.space.axes)
        if len(self.vertices) < dimension + 1:
            raise ValueError(
                "a full-dimensional V-polytope needs at least dimension + 1 vertices"
            )
        vertex_ids = tuple(vertex.vertex_id for vertex in self.vertices)
        if tuple(sorted(vertex_ids)) != vertex_ids or len(set(vertex_ids)) != len(
            vertex_ids
        ):
            raise ValueError("vertex IDs must be unique and strictly ordered")
        coordinates = tuple(vertex.coordinates for vertex in self.vertices)
        if any(len(point) != dimension for point in coordinates):
            raise ValueError("every vertex must use the polytope coordinate axis")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("polytope vertices must have distinct coordinates")
        from jacobian.math.polytope._operations import (
            require_full_dimensional_extreme_vertices,
        )

        require_full_dimensional_extreme_vertices(self)
        return self


class RationalCovector(StrictModel):
    """An exact covector paired with one labelled rational coordinate space."""

    space: RationalCoordinateSpace
    components: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_DIMENSION,
        description=(
            "Exact covector components in the declared coordinate-axis order, "
            "each one canonical reduced rational."
        ),
    )

    @model_validator(mode="after")
    def require_components_match_declared_axis(self) -> Self:
        if len(self.components) != len(self.space.axes):
            raise ValueError(
                "covector components must use the declared coordinate axis"
            )
        return self


class RationalExposedFace(StrictModel):
    """The complete vertex family of one exposed face of a V-polytope."""

    space: RationalCoordinateSpace
    vertices: tuple[RationalPolytopeVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
    )

    @model_validator(mode="after")
    def require_canonical_face_vertices(self) -> Self:
        vertex_ids = tuple(vertex.vertex_id for vertex in self.vertices)
        if tuple(sorted(vertex_ids)) != vertex_ids or len(set(vertex_ids)) != len(
            vertex_ids
        ):
            raise ValueError(
                "exposed-face vertex IDs must be unique and strictly ordered"
            )
        dimension = len(self.space.axes)
        if any(len(vertex.coordinates) != dimension for vertex in self.vertices):
            raise ValueError(
                "every exposed-face vertex must use the face coordinate axis"
            )
        if len({vertex.coordinates for vertex in self.vertices}) != len(self.vertices):
            raise ValueError("exposed-face vertices must have distinct coordinates")
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
        max_length=MAX_DIMENSION,
        description=(
            "Normal vector a of the half-space <a, x> <= b; at least one "
            "entry must be nonzero (all-zero rows are rejected)."
        ),
    )
    offset: CanonicalRational


def require_support_components_within_envelope(
    polytope: RationalVPolytope,
    covector: RationalCovector,
) -> None:
    """Enforce the support operation's per-component execution envelope.

    Canonical polytope values admit every canonical rational coordinate;
    this smaller operation-specific bound is the single admission decision
    shared by the support request model and the native ``polytope_support``
    entry point, keeping the exact hull intermediates of one accepted call
    inside the bounded envelope derived for ``MAX_SUPPORT_COMPONENT_DIGITS``.
    """

    for vertex in polytope.vertices:
        for coordinate in vertex.coordinates:
            require_bounded_rational(
                coordinate,
                max_digits=MAX_SUPPORT_COMPONENT_DIGITS,
                label="polytope vertex coordinate",
            )
    for component in covector.components:
        require_bounded_rational(
            component,
            max_digits=MAX_SUPPORT_COMPONENT_DIGITS,
            label="covector component",
        )


class PolytopeSupportRequest(StrictModel):
    """Compute one support value and its complete exposed vertex face.

    The input polytope is full-dimensional and already carries its complete
    exact V-representation. Evaluation then performs one deterministic
    ``O(n*d)`` rational dot-product pass; no H/V conversion or optimization
    solver is introduced by this operation.

    The polytope and the covector must declare one common labelled
    coordinate space: their serialized ``space`` values must be identical
    (same axis labels in the same order), and mismatched spaces are rejected
    before any evaluation. Each vertex coordinate and covector component is
    a canonical rational carrying at most 150 digits per reduced numerator
    or denominator — an operation-specific envelope, stricter than the
    global canonical limit.
    """

    polytope: RationalVPolytope = Field(
        description=(
            "Full-dimensional exact V-polytope whose serialized ``space`` "
            "(axis labels and order) must be identical to the covector's "
            "serialized ``space``; every vertex coordinate carries at most "
            f"{MAX_SUPPORT_COMPONENT_DIGITS} digits per reduced numerator "
            "or denominator."
        )
    )
    covector: RationalCovector = Field(
        description=(
            "Exact covector whose serialized ``space`` (axis labels and "
            "order) must be identical to the polytope's serialized "
            "``space``; every component carries at most "
            f"{MAX_SUPPORT_COMPONENT_DIGITS} digits per reduced numerator "
            "or denominator."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_components_within_support_envelope(cls, data: object) -> object:
        """Reject over-envelope components before nested V-polytope parsing.

        Pydantic constructs (and canonically proves) nested values before
        parent after-validators run, so a raw ``math.run`` payload whose
        components sit between this operation's 150-digit envelope and the
        global canonical limit would reach the exact hull-facet proof
        inside ``RationalVPolytope`` construction before
        ``require_admitted_support_components`` could reject it. This
        preflight measures only the authored reduced components of the raw
        payload — dict or already-built values alike — so even a rejected
        request stays inside the advertised execution envelope; the
        canonical V-polytope value's broader domain is unchanged.
        """

        return _preflight_raw_support_components(data)

    @model_validator(mode="after")
    def require_common_coordinate_space(self) -> Self:
        if self.polytope.space != self.covector.space:
            raise ValueError("polytope and covector must use the same coordinate space")
        return self

    @model_validator(mode="after")
    def require_admitted_support_components(self) -> Self:
        require_support_components_within_envelope(self.polytope, self.covector)
        return self


class PolytopeSupportResult(StrictModel):
    """A source-bound exact support value and its complete exposed face.

    The retained source satisfies the same admitted execution envelope as
    ``PolytopeSupportRequest``: each polytope vertex coordinate and
    covector component carries at most 150 digits per reduced numerator or
    denominator, so deserialization accepts only outputs the admitted
    operation can produce and never replays a source outside the envelope.
    """

    polytope: RationalVPolytope
    covector: RationalCovector
    support_value: CanonicalRational
    exposed_face: RationalExposedFace

    @model_validator(mode="before")
    @classmethod
    def require_raw_components_within_support_envelope(cls, data: object) -> object:
        """Preflight the retained source before nested V-polytope parsing.

        Deserializing a serialized result constructs (and canonically
        proves) the nested values before parent after-validators run, so
        the same raw-payload measurement as the request rejects an
        over-envelope retained source before the exact hull-facet proof
        can execute outside the advertised envelope.
        """

        return _preflight_raw_support_components(data)

    @model_validator(mode="after")
    def bind_support_data_to_source(self) -> Self:
        if self.polytope.space != self.covector.space:
            raise ValueError("polytope and covector must use the same coordinate space")
        require_support_components_within_envelope(self.polytope, self.covector)
        from jacobian.math.polytope._operations import support_data

        expected_value, expected_vertices = support_data(self.polytope, self.covector)
        if self.support_value != CanonicalRational.from_fraction(expected_value):
            raise ValueError(
                "support value must equal the exact maximum on every vertex"
            )
        expected_face = RationalExposedFace(
            space=self.polytope.space,
            vertices=expected_vertices,
        )
        if self.exposed_face != expected_face:
            raise ValueError(
                "exposed face must be exactly the complete maximizing vertex family"
            )
        return self


def _canonical_v_polytope_vertices(polytope: RationalVPolytope) -> tuple[Vertex, ...]:
    """Map the labelled canonical V-polytope onto bare volume vertices.

    The labelled coordinate space fixes the axis order, so vertex
    coordinates are carried over positionally and unchanged.
    """

    return tuple(Vertex(coordinates=vertex.coordinates) for vertex in polytope.vertices)


def _v_polytope_axis_count(value: object) -> int | None:
    """Return the declared ambient dimension of one V-polytope payload.

    Unrecognized payload shapes return ``None`` so ordinary canonical
    validation reports them with the published schema errors.
    """

    if isinstance(value, RationalVPolytope):
        return len(value.space.axes)
    if isinstance(value, dict):
        space = value.get("space")
        if isinstance(space, RationalCoordinateSpace):
            return len(space.axes)
        if isinstance(space, dict) and isinstance(space.get("axes"), (list, tuple)):
            return len(space["axes"])
    return None


def _require_projected_dimension_bound(dimension: int, dimension_bound: object) -> None:
    """Reject a V-polytope outside the published bound before hull replay.

    The raw bound is measured with the ``dimension_bound`` field's own
    schema, derived from its declaration so the accepted coercion domain
    cannot drift: malformed, null, and out-of-range values are rejected
    here with the accept/reject boundary ordinary field validation
    enforces, before the canonical reconstruction replays the exact
    extremality proof, while a coercible raw value is compared through
    its coerced integer exactly as the outer model would.
    """

    if dimension_bound is None:
        raise ValueError(
            f"dimension_bound must be an integer between 1 and {MAX_DIMENSION}"
        )
    try:
        bound: int = _DIMENSION_BOUND_ADAPTER.validate_python(dimension_bound)
    except ValidationError as exc:
        raise ValueError(
            f"dimension_bound must be an integer between 1 and {MAX_DIMENSION}"
        ) from exc
    if dimension > bound:
        raise ValueError(f"dimension {dimension} exceeds the dimension bound {bound}")


VertexTuple = Annotated[
    tuple[Vertex, ...],
    Field(min_length=1, max_length=MAX_VERTICES),
]


def _tuple_canonical_containers(value: Any) -> Any:
    """Return raw JSON payloads with every sequence materialized as a tuple.

    Dispatch parses each request through strict JSON validation, so a
    preflight validator that hands back raw JSON arrays would make the
    strict parser reject list-to-tuple coercion on canonical tuple fields.
    Recursively converting containers keeps the preflight semantics while
    preserving the declared canonical shapes; the payload size is already
    bounded by the request's own container limits.
    """

    if isinstance(value, list):
        return tuple(_tuple_canonical_containers(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuple_canonical_containers(item) for key, item in value.items()}
    return value


class PolytopeVolumeRequest(StrictModel):
    """A bounded rational polytope in exactly one of the two representations.

    The V-representation is given either as bare vertices or unchanged as
    the domain's canonical labelled ``RationalVPolytope`` value (for
    example the ``polytope`` of a support result), constructed or
    serialized; admission enforces the same work bound on both forms.

    Admission enforces a work bound that couples vertex count with ambient
    dimension: after duplicate points are removed, the exact hull
    enumeration considers ``C(n, d)`` d-subsets of ``n`` distinct vertices
    in dimension ``d``, and requests with ``C(n, d) > 200000`` are rejected.
    The same limit applies to the vertex set derived from an
    H-representation, whose own rows are additionally bounded by
    ``C(m, d) <= 700000`` on the distinct half-spaces (see the field
    descriptions for the exact published rules).
    """

    vertices: VertexTuple | RationalVPolytope | None = Field(
        default=None,
        description=(
            "V-representation: the vertices of the convex hull, either as "
            "bare coordinate vertices or as one canonical labelled "
            "``RationalVPolytope`` value (its serialized ``space``/"
            "``vertices`` shape is accepted too), such as the ``polytope`` "
            "of a support result. "
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

    @model_validator(mode="before")
    @classmethod
    def accept_canonical_v_polytope_value(cls, data: object) -> object:
        """Project the canonical labelled V-polytope onto bare vertices.

        Support results carry ``RationalVPolytope`` as the domain's
        canonical V-polytope value, so composing one into a volume request
        must not force callers to discard the labelled space and rebuild
        every vertex. Both the constructed value and its serialized
        ``space``/``vertices`` shape are accepted unchanged and mapped
        positionally (the labelled axis fixes the coordinate order) before
        ordinary validation, so admission and the kernel see exactly the
        declared V-representation; a serialized value is re-validated as
        the canonical type first.  Reconstructing that type replays its
        exact extremality proof, so every cheap outer field — the closed
        field set, the halfspace conflict, and the whole published
        ``dimension_bound`` schema — is preflighted first: an
        already-invalid request must fail before any hull replay runs.
        """

        if not isinstance(data, dict):
            return data
        value = data.get("vertices")
        carries_v_polytope = isinstance(value, RationalVPolytope) or (
            isinstance(value, dict) and set(value) == {"space", "vertices"}
        )
        if carries_v_polytope:
            unknown_fields = set(data) - {"vertices", "halfspaces", "dimension_bound"}
            if unknown_fields:
                raise ValueError(
                    "unexpected fields for a polytope volume request: "
                    f"{sorted(unknown_fields)}"
                )
            if data.get("halfspaces") is not None:
                raise ValueError(
                    "exactly one of `vertices` or `halfspaces` must be provided"
                )
            axis_count = _v_polytope_axis_count(value)
            if axis_count is not None:
                _require_projected_dimension_bound(
                    axis_count, data.get("dimension_bound", MAX_DIMENSION)
                )
        if isinstance(value, RationalVPolytope):
            return {**data, "vertices": _canonical_v_polytope_vertices(value)}
        if isinstance(value, dict) and set(value) == {"space", "vertices"}:
            canonical = RationalVPolytope.model_validate(value)
            return {**data, "vertices": _canonical_v_polytope_vertices(canonical)}
        return _tuple_canonical_containers(data)

    @model_validator(mode="after")
    def validate_representation(self) -> Self:
        has_v = self.vertices is not None
        has_h = self.halfspaces is not None
        if has_v == has_h:
            raise ValueError(
                "exactly one of `vertices` or `halfspaces` must be provided"
            )
        if has_v:
            vertices = self.vertices
            assert isinstance(vertices, tuple)  # the before-validator projects it
            _validate_vertices(vertices, self.dimension_bound)
        else:
            assert self.halfspaces is not None  # for type checkers
            _validate_halfspaces(self.halfspaces, self.dimension_bound)
        return self


_DIMENSION_BOUND_ADAPTER: TypeAdapter[int] = TypeAdapter(
    Annotated[int, *PolytopeVolumeRequest.model_fields["dimension_bound"].metadata]
)


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
    "MAX_DIMENSION",
    "MAX_FACETS",
    "MAX_HULL_SUBFACETS",
    "MAX_SUPPORT_COMPONENT_DIGITS",
    "MAX_SUPPORT_ORIENTATION_TESTS",
    "MAX_SUPPORT_VERTEX_SUBSETS",
    "MAX_VERTICES",
    "Halfspace",
    "PolytopeSupportRequest",
    "PolytopeSupportResult",
    "PolytopeVolumeRequest",
    "PolytopeVolumeResult",
    "RationalCoordinateSpace",
    "RationalCovector",
    "RationalExposedFace",
    "RationalPolytopeVertex",
    "RationalVPolytope",
    "Vertex",
]
