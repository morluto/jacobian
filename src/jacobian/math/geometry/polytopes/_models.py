"""Request/result models for the bounded rational polytope domain."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from fractions import Fraction
from itertools import combinations
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError
from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.geometry.polytopes._rational_geometry import (
    recession_cone_is_trivial,
    vertices_from_halfspaces,
)
from jacobian.math.geometry.polytopes.values import (
    MAX_RATIONAL_POLYTOPE_DIMENSION,
    Halfspace,
    Vertex,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Create a stable structured error for the polytope public contract."""

    return PydanticCustomError(f"polytope.{reason}", message)


class PolytopeAdmissionError(ValueError):
    """Native admission failure for polytope volume operations."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


MAX_DIMENSION = 6
"""Ambient-dimension bound shared by the volume and support operations.

Exact volume caps ambient dimension here, and support pairs a V-polytope
with a covector of at most this many components. The canonical labelled
V-representation itself may carry up to ``MAX_FACET_DIMENSION`` axes so its
consumer path covers the facet profiles' wider published domain; consumers
whose own envelopes are narrower reject over-dimensional values through
their published dimension bounds.
"""

MAX_FACET_DIMENSION = MAX_RATIONAL_POLYTOPE_DIMENSION
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

MAX_COORDINATE_LABEL_LENGTH = 64
"""Maximum Unicode-scalar length of an axis or vertex identifier."""

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

MAX_SUBSYSTEM_SOLVES = 5_000_000
"""Absolute combinatorial ceiling for vertex enumeration from H-representations."""


def _largest_combination_axis(*, ceiling: int, dimension: int, work: int) -> int:
    """Largest axis size whose exact combination count fits ``work``."""

    size = ceiling
    while math.comb(size, dimension) > work:
        size -= 1
    return size


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
inputs and the exact construction work required by the public
V-polytope value.
"""

MAX_SUPPORT_VERTEX_SUBSETS = 100_000
"""Maximum ``C(n, d)`` exact subfacets used to establish a V-polytope.

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

MAX_EXTREMALITY_HEIGHT_WORK = 20_000_000_000
"""Ceiling coupling extremality-proof work with coordinate height.

The V-polytope extremality proof evaluates ``T = C(n, d) * (n - d)``
orientation determinants on ``(d+1) x (d+1)`` rational matrices. With ``D``
the largest decimal digit count over every reduced numerator and denominator,
clearing each row's denominators bounds matrix entries by ``(d + 2) * D``
digits, Hadamard's bound bounds every fraction-free elimination intermediate
by ``(d + 1) * (d + 2) * D`` digits, and the elimination performs
``O((d + 1)^3)`` multiplications of such operands. With ``d <= MAX_FACET_DIMENSION``
fixed, one orientation test therefore costs ``Theta(D^2)`` limb operations,
and the complete proof ``T * D^2`` up to constants absorbed by this ceiling.
The ceiling is calibrated above the worst work the published operation
envelopes admit (the support regime: 500,000 tests at 150 digits gives
1.125e10), so values inside those envelopes validate unchanged while larger
heights grade the admitted test count down proportionally — 19,073 tests at
1,024 digits, 18 at the 32,768-digit canonical limit.
"""


def _require_unicode_scalar_label(value: str) -> str:
    """Reject labels carrying code points strict JSON cannot encode.

    Unpaired surrogates are not Unicode scalar values, so RFC 8785
    serialization of an accepted value containing one would fail at the
    supported transport boundary; they are outside the admitted label
    domain.
    """

    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "labels must contain only Unicode scalar values"
        )
    return value


CoordinateAxis = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_COORDINATE_LABEL_LENGTH,
        strict=True,
    ),
    AfterValidator(_require_unicode_scalar_label),
]
"""One coordinate identifier in an ordered labelled rational space.

Axis labels must be Unicode scalar strings: unpaired surrogates cannot be
encoded by the domain's strict JSON transport.
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
    components sit between the operation's published component envelope and the
    global canonical limit is rejected before nested V-polytope parsing
    constructs (and canonically proves) the retained source.
    """

    if not isinstance(data, dict):
        return data
    canonical: Any = canonicalize_json_containers(data)
    _require_raw_support_covector_admissible(canonical)
    for vertex in _iter_raw_entries(canonical.get("polytope"), "vertices"):
        for component in _iter_raw_entries(vertex, "coordinates"):
            _require_raw_component_within_support_envelope(
                component,
                "polytope vertex coordinate",
            )
            _require_raw_canonical_rational_component(
                component,
                "polytope vertex coordinate",
            )
    for component in _iter_raw_entries(canonical.get("covector"), "components"):
        _require_raw_component_within_support_envelope(
            component,
            "covector component",
        )
    return canonical


def _require_raw_component_digit_bound(
    component: object,
    label: str,
    max_digits: int,
) -> None:
    """Measure one authored rational payload against a per-component bound.

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
    if max(len(num.lstrip("-")), len(den.lstrip("-"))) > max_digits:
        raise _validation_error(
            "component_digit_bound", f"{label} exceeds the {max_digits}-digit bound"
        )


def _require_raw_component_within_support_envelope(
    component: object,
    label: str,
) -> None:
    """Measure one authored rational payload against the support envelope."""

    _require_raw_component_digit_bound(
        component,
        label,
        MAX_SUPPORT_COMPONENT_DIGITS,
    )


def _require_raw_v_polytope_coordinates_within_facet_envelope(value: object) -> None:
    """Measure authored V-polytope coordinates against the facet envelope.

    The facet operation admits at most ``MAX_FACET_COORDINATE_DIGITS`` digits
    per reduced coordinate component. Measuring the authored coordinates here
    rejects an incompatible composed value before nested canonical parsing;
    unrecognized shapes fall through to ordinary structural errors.
    """

    for vertex in _iter_raw_entries(value, "vertices"):
        for component in _iter_raw_entries(vertex, "coordinates"):
            _require_raw_component_digit_bound(
                component,
                "facet-profile vertex coordinate",
                MAX_FACET_COORDINATE_DIGITS,
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
    object or an already-built value, so this gate constructs one from
    every raw ``{num, den}`` payload: the strict canonical-integer
    grammar, the global digit bound, and the reduced positive-denominator
    invariant are all enforced here, before nested validation would run
    them only after the polytope field has paid its exact hull proof.
    Any other authored shape is certain to be rejected by nested
    validation too, so it raises immediately.
    """

    if isinstance(component, CanonicalRational):
        return
    if (
        isinstance(component, dict)
        and set(component) == {"num", "den"}
        and isinstance(component["num"], str)
        and isinstance(component["den"], str)
    ):
        try:
            CanonicalRational.model_validate(component)
        except ValidationError as exc:
            raise _validation_error(
                "canonical_rational", f"{label} must be a canonical rational"
            ) from exc
        return
    raise _validation_error(
        "canonical_rational", f"{label} must be a canonical rational"
    )


def _require_raw_coordinate_space(value: object, label: str) -> tuple[str, ...]:
    """Mirror ``RationalCoordinateSpace`` on one raw payload value.

    Returns the declared axis labels when the raw space satisfies every
    published constraint (closed ``axes`` field, non-empty sequence of at
    most ``MAX_FACET_DIMENSION`` short unique string labels); any violation
    raises here because ordinary nested validation rejects it too, only
    after the hull proof has run.
    """

    if isinstance(value, RationalCoordinateSpace):
        return tuple(value.axes)
    if not isinstance(value, dict) or set(value) != {"axes"}:
        raise _validation_error(
            "coordinate_space_shape", f"{label} space must be an object with axes"
        )
    axes = value["axes"]
    if not isinstance(axes, (list, tuple)) or not axes:
        raise _validation_error(
            "coordinate_space_shape", f"{label} space axes must be a non-empty sequence"
        )
    if len(axes) > MAX_FACET_DIMENSION:
        raise _validation_error(
            "coordinate_space_shape",
            f"{label} space must declare at most {MAX_FACET_DIMENSION} axes",
        )
    if any(
        not isinstance(axis, str) or not 1 <= len(axis) <= MAX_COORDINATE_LABEL_LENGTH
        for axis in axes
    ):
        raise _validation_error(
            "coordinate_space_axes", f"{label} space axes must be short string labels"
        )
    for axis in axes:
        _require_unicode_scalar_label(axis)
    if len(set(axes)) != len(axes):
        raise _validation_error(
            "coordinate_axes_unique", "coordinate axes must be unique"
        )
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
    shapes, the declared-axis match, and space agreement with the
    polytope — including an already-built covector whose space disagrees
    with the raw polytope's.
    """

    covector = canonical.get("covector")
    if covector is None:
        raise _validation_error("covector_required", "covector must be provided")
    if isinstance(covector, RationalCovector):
        axes = tuple(covector.space.axes)
    else:
        if not isinstance(covector, dict) or set(covector) != {
            "space",
            "components",
        }:
            raise _validation_error(
                "covector_shape", "covector must be an object with space and components"
            )
        components = covector["components"]
        if not isinstance(components, (list, tuple)):
            raise _validation_error(
                "covector_components", "covector components must be a sequence"
            )
        if not components:
            raise _validation_error(
                "covector_components",
                "covector components must be a non-empty sequence",
            )
        if len(components) > MAX_DIMENSION:
            raise _validation_error(
                "covector_components",
                f"covector components must carry at most {MAX_DIMENSION} entries",
            )
        for component in components:
            _require_raw_canonical_rational_component(component, "covector component")
        axes = _require_raw_coordinate_space(covector["space"], "covector")
        if len(components) != len(axes):
            raise _validation_error(
                "covector_components",
                "covector components must use the declared coordinate axis",
            )
    polytope_axes = _raw_space_axes(
        _raw_field_value(canonical.get("polytope"), "space")
    )
    if (
        polytope_axes is not None
        and all(isinstance(axis, str) for axis in polytope_axes)
        and tuple(polytope_axes) != axes
    ):
        raise _validation_error(
            "coordinate_space_mismatch",
            "polytope and covector must use the same coordinate space",
        )


def _require_raw_exposed_face_vertex(
    vertex: object,
    dimension: int,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Mirror one ``RationalPolytopeVertex`` entry of a raw exposed face.

    Returns the vertex ID and its reduced-component key so the caller can
    apply the exposed face's defining ordering and distinctness
    invariants; any shape outside the published schema raises here,
    including a vertex ID outside the Unicode scalar label grammar.
    """

    if isinstance(vertex, RationalPolytopeVertex):
        return vertex.vertex_id, tuple(
            (component.num, component.den) for component in vertex.coordinates
        )
    if (
        not isinstance(vertex, dict)
        or set(vertex) != {"vertex_id", "coordinates"}
        or not isinstance(vertex["vertex_id"], str)
        or not 1 <= len(vertex["vertex_id"]) <= MAX_COORDINATE_LABEL_LENGTH
    ):
        raise _validation_error(
            "exposed_face_vertex_shape",
            "exposed face vertex must be an object with a short vertex_id "
            "and coordinates",
        )
    _require_unicode_scalar_label(vertex["vertex_id"])
    coordinates = vertex["coordinates"]
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise _validation_error(
            "exposed_face_coordinates",
            "exposed face vertex coordinates must be a non-empty sequence",
        )
    if len(coordinates) != dimension:
        raise _validation_error(
            "exposed_face_vertex",
            "every exposed-face vertex must use the face coordinate axis",
        )
    for component in coordinates:
        _require_raw_canonical_rational_component(
            component, "exposed face vertex coordinate"
        )
    return vertex["vertex_id"], tuple(
        (component.num, component.den)
        if isinstance(component, CanonicalRational)
        else (component["num"], component["den"])
        for component in coordinates
    )


def _require_raw_support_request_shape(canonical: Any) -> None:
    """Gate the request's closed field set before V-polytope parsing.

    Pydantic reports forbidden extra fields only after every declared
    field has been parsed and aggregated, so a raw payload with a valid
    near-limit source pays the exact hull proof before ``StrictModel``
    rejects the extra key. Only the published ``{polytope, covector}``
    field set is mirrored here, with the accept/reject boundary ordinary
    validation enforces.
    """

    if not isinstance(canonical, dict):
        return
    unknown_fields = set(canonical) - {"polytope", "covector"}
    if unknown_fields:
        raise _validation_error(
            "unexpected_fields",
            "unexpected fields for a polytope support request: "
            f"{sorted(unknown_fields)}",
        )


def _require_raw_support_conclusions_admissible(canonical: Any) -> None:
    """Gate the outer shape and conclusion fields of a raw support result.

    Pydantic parses declared fields in order, so deserializing a result
    whose retained source is valid near the hull envelope pays the exact
    extremality proof before a missing or malformed ``support_value`` or
    ``exposed_face``, or a forbidden extra field, is reported. This gate
    mirrors only the result-level constraints nested validation rejects
    anyway: the closed field set and the published shape of both
    conclusion fields, including the exposed face's defining ordering
    and distinctness invariants and its space agreement with the
    retained polytope — including an already-built face whose space
    disagrees with the raw polytope's.
    """

    if not isinstance(canonical, dict):
        return
    unknown_fields = set(canonical) - {
        "polytope",
        "covector",
        "support_value",
        "exposed_face",
    }
    if unknown_fields:
        raise _validation_error(
            "unexpected_fields",
            f"unexpected fields for a support result: {sorted(unknown_fields)}",
        )

    support_value = canonical.get("support_value")
    if support_value is None:
        raise _validation_error(
            "support_value_binding", "support_value must be provided"
        )
    _require_raw_canonical_rational_component(support_value, "support value")

    exposed_face = canonical.get("exposed_face")
    if exposed_face is None:
        raise _validation_error(
            "exposed_face_required", "exposed_face must be provided"
        )
    polytope_axes = _raw_space_axes(
        _raw_field_value(canonical.get("polytope"), "space")
    )
    if isinstance(exposed_face, RationalExposedFace):
        axes = tuple(exposed_face.space.axes)
    else:
        if not isinstance(exposed_face, dict) or set(exposed_face) != {
            "space",
            "vertices",
        }:
            raise _validation_error(
                "exposed_face_shape",
                "exposed face must be an object with space and vertices",
            )
        axes = _require_raw_coordinate_space(exposed_face["space"], "exposed face")
    if (
        polytope_axes is not None
        and all(isinstance(axis, str) for axis in polytope_axes)
        and tuple(polytope_axes) != axes
    ):
        raise _validation_error(
            "coordinate_space_mismatch",
            "exposed face must use the same coordinate space as the polytope",
        )
    if isinstance(exposed_face, RationalExposedFace):
        # Its ordering, distinctness, and serialization invariants already
        # hold by construction. Whether it is the complete maximizing face is
        # established by the owning support operation rather than while
        # parsing a result.
        return
    vertices = exposed_face["vertices"]
    if not isinstance(vertices, (list, tuple)):
        raise _validation_error(
            "exposed_face_vertices", "exposed face vertices must be a sequence"
        )
    if not vertices:
        raise _validation_error(
            "exposed_face_vertices",
            "exposed face vertices must be a non-empty sequence",
        )
    if len(vertices) > MAX_VERTICES:
        raise _validation_error(
            "exposed_face_vertices",
            f"exposed face vertices must carry at most {MAX_VERTICES} entries",
        )
    parsed = [
        _require_raw_exposed_face_vertex(vertex, len(axes)) for vertex in vertices
    ]
    vertex_ids = tuple(vertex_id for vertex_id, _ in parsed)
    if vertex_ids != tuple(sorted(vertex_ids)) or len(set(vertex_ids)) != len(
        vertex_ids
    ):
        raise _validation_error(
            "exposed_face_vertex",
            "exposed-face vertex IDs must be unique and strictly ordered",
        )
    coordinate_rows = tuple(rows for _, rows in parsed)
    if len(set(coordinate_rows)) != len(coordinate_rows):
        raise _validation_error(
            "exposed_face_coordinates_unique",
            "exposed-face vertices must have distinct coordinates",
        )


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
        raise PolytopeAdmissionError(
            "volume_result_bound",
            "coordinate magnitudes can grow the exact volume beyond the "
            f"{MAX_RESULT_COMPONENT_DIGITS}-digit canonical rational "
            "result bound",
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
        raise PolytopeAdmissionError(
            "volume_result_bound",
            "coordinate magnitudes can grow the exact volume beyond the "
            f"{MAX_RESULT_COMPONENT_DIGITS}-digit canonical rational "
            "result bound",
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


def _hull_subfacets(points: list[list[Rational]], dim: int) -> list[tuple[int, ...]]:
    """Enumerate the dim-subsets of points on the convex hull boundary.

    A dim-subset is a (d-1)-subfacet if all remaining points lie on one
    side (or on) the hyperplane it spans. Subfacets of a coplanar larger
    facet are returned individually; merge with ``_max_facets``.
    """

    n = len(points)
    subfacets: list[tuple[int, ...]] = []
    for subset in combinations(range(n), dim):
        signs: set[int] = set()
        ok = True
        for p in range(n):
            if p in subset:
                continue
            mat = Matrix(
                [[points[i][k] for k in range(dim)] + [1] for i in subset]
                + [[points[p][k] for k in range(dim)] + [1]]
            )
            det = mat.det()
            if det > 0:
                signs.add(1)
            elif det < 0:
                signs.add(-1)
            if len(signs) > 1:
                ok = False
                break
        if ok and signs:
            subfacets.append(tuple(subset))
    return subfacets


def _plane_signature(
    subfacet: tuple[int, ...], points: list[list[Rational]]
) -> tuple[int, ...] | None:
    """Canonical signature of the hyperplane through the subfacet points."""

    dim = len(subfacet)
    mat = Matrix([[points[i][k] for k in range(dim)] + [1] for i in subfacet])
    nullspace = mat.nullspace()
    if not nullspace:
        return None
    vec = [Rational(nullspace[0][j]) for j in range(dim + 1)]
    first_nonzero = next(j for j in range(dim + 1) if vec[j] != 0)
    sign = 1 if vec[first_nonzero] > 0 else -1
    denominators = [v.denominator for v in vec]
    lcm = 1
    for denominator in denominators:
        lcm = lcm * denominator // math.gcd(lcm, denominator)
    scaled = [int(v * sign * lcm) for v in vec]
    gcd = 0
    for value in scaled:
        gcd = math.gcd(gcd, abs(value))
    if gcd == 0:
        gcd = 1
    return tuple(value // gcd for value in scaled)


def _max_facets(points: list[list[Rational]], dim: int) -> list[list[int]]:
    """Return the maximal (d-1)-facets as sorted index lists."""

    subfacets = _hull_subfacets(points, dim)
    groups: dict[tuple[int, ...], set[int]] = {}
    for subfacet in subfacets:
        sig = _plane_signature(subfacet, points)
        if sig is None:
            continue
        groups.setdefault(sig, set()).update(subfacet)
    return [sorted(members) for members in groups.values()]


def _extreme_point_indices(
    groups: dict[tuple[int, ...], set[int]],
    point_count: int,
    dim: int,
) -> tuple[list[int], list[int]]:
    """Return (extreme indices, boundary counts) from grouped maximal facets."""

    counts = [0] * point_count
    active_normals: list[list[list[Rational]]] = [[] for _ in range(point_count)]
    for normal, members in groups.items():
        normal_values = list(normal[:-1])
        for index in members:
            if 0 <= index < point_count:
                counts[index] += 1
                active_normals[index].append(normal_values)
    kept = [
        index
        for index in range(point_count)
        if active_normals[index] and Matrix(active_normals[index]).rank() == dim
    ]
    return kept, counts


def _filter_redundant_vertices(
    points: list[list[Rational]], dim: int
) -> list[list[Rational]]:
    """Return extreme hull vertices, dropping redundant boundary points."""

    if len(points) <= dim + 1:
        return points
    subfacet_count = math.comb(len(points), dim)
    if subfacet_count > MAX_HULL_SUBFACETS:
        return points
    groups: dict[tuple[int, ...], set[int]] = {}
    for subfacet in _hull_subfacets(points, dim):
        sig = _plane_signature(subfacet, points)
        if sig is None:
            continue
        groups.setdefault(sig, set()).update(subfacet)
    if not groups:
        return points
    keep_indices, counts = _extreme_point_indices(groups, len(points), dim)
    if len(keep_indices) < dim + 1:
        hull_indices = [index for index, count in enumerate(counts) if count > 0]
        if len(hull_indices) >= dim + 1:
            keep_indices = hull_indices
        else:
            return points
    keep_set = set(keep_indices)
    return [point for index, point in enumerate(points) if index in keep_set]


def _project_facet(
    facet_points: list[list[Rational]], dim: int
) -> list[list[Rational]]:
    """Project a coplanar dim-dim facet into (dim-1)-dim coordinates."""

    for axis in range(dim):
        projected = [
            [point[k] for k in range(dim) if k != axis] for point in facet_points
        ]
        if _rank_of_diffs(projected, dim - 1) == dim - 1:
            return projected
    return [[point[k] for k in range(dim - 1)] for point in facet_points]


def _triangulate_2d(points: list[list[Rational]]) -> list[tuple[int, ...]]:
    """Triangulate a 2D convex polygon by a fan from its first corner."""

    subfacets = _hull_subfacets(points, 2)
    adjacency: dict[int, set[int]] = {}
    for edge in subfacets:
        adjacency.setdefault(edge[0], set()).add(edge[1])
        adjacency.setdefault(edge[1], set()).add(edge[0])
    corners = [index for index, neighbors in adjacency.items() if len(neighbors) == 2]
    if not corners:
        return []
    start = corners[0]
    order = [start]
    previous = -1
    current = start
    while True:
        neighbors = [value for value in adjacency[current] if value != previous]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt == start:
            break
        order.append(nxt)
        previous, current = current, nxt
        if len(order) > len(corners) + 1:
            break
    return [
        (order[0], order[index], order[index + 1]) for index in range(1, len(order) - 1)
    ]


def _triangulate(points: list[list[Rational]], dim: int) -> list[tuple[int, ...]]:
    """Return a triangulation of the convex hull as simplicial index tuples."""

    n = len(points)
    if n < dim + 1:
        return []
    if dim == 1:
        coordinates = sorted({point[0] for point in points})
        if len(coordinates) < 2:
            return []
        minimum = min(range(n), key=lambda index: points[index][0])
        maximum = max(range(n), key=lambda index: points[index][0])
        return [(minimum, maximum)]
    if dim == 2:
        return _triangulate_2d(points)
    apex = _extreme_vertex(points, dim)
    if apex is None:
        return []
    facets = _max_facets(points, dim)
    triangulation: list[tuple[int, ...]] = []
    for members in facets:
        if apex in members:
            continue
        facet_points = [points[index] for index in members]
        projected = _project_facet(facet_points, dim)
        projected_triangulation = _triangulate(projected, dim - 1)
        for tri in projected_triangulation:
            triangulation.append((*tuple(members[index] for index in tri), apex))
    return triangulation


def _rank_of_diffs(points: list[list[Rational]], dim: int) -> int:
    """Rank of the matrix of ``point - point[0]`` differences in ``dim`` dims."""

    if len(points) <= 1:
        return 0
    reference = points[0]
    columns = [
        Matrix([[points[index][axis] - reference[axis]] for axis in range(dim)])
        for index in range(1, len(points))
    ]
    return Matrix.hstack(*columns).rank() if columns else 0


def _extreme_vertex(points: list[list[Rational]], dim: int) -> int | None:
    """Return the index of one extreme hull vertex."""

    subfacets = _hull_subfacets(points, dim)
    if not subfacets:
        return None
    on_hull: set[int] = set()
    for subfacet in subfacets:
        on_hull.update(subfacet)
    for index in range(len(points)):
        if index in on_hull:
            return index
    return None


def _deduplicate_halfspaces(halfspaces: tuple[Halfspace, ...]) -> tuple[Halfspace, ...]:
    """Drop duplicate half-spaces up to positive scaling."""

    seen: set[tuple[tuple[int, ...], tuple[int, int]]] = set()
    unique: list[Halfspace] = []
    for halfspace in halfspaces:
        fractions = [
            Fraction(*coefficient.as_integer_ratio())
            for coefficient in halfspace.coefficients
        ]
        offset = Fraction(*halfspace.offset.as_integer_ratio())
        lcm = 1
        for fraction in fractions:
            lcm = lcm * fraction.denominator // math.gcd(lcm, fraction.denominator)
        ints = [int(fraction * lcm) for fraction in fractions]
        gcd = 0
        for integer in ints:
            gcd = math.gcd(gcd, abs(integer))
        normalized = (
            tuple(integer // gcd for integer in ints),
            ((offset * lcm / gcd).numerator, (offset * lcm / gcd).denominator),
        )
        if normalized not in seen:
            seen.add(normalized)
            unique.append(halfspace)
    return tuple(unique)


def _halfspace_rows(
    halfspaces: tuple[Halfspace, ...],
) -> list[tuple[list[Rational], Rational]]:
    """Convert halfspaces to rational coefficient/offset rows."""

    return [
        (
            [
                Rational(*coefficient.as_integer_ratio())
                for coefficient in hs.coefficients
            ],
            Rational(*hs.offset.as_integer_ratio()),
        )
        for hs in halfspaces
    ]


def _vertices_from_v_representation(
    vertices: tuple[Vertex, ...],
) -> tuple[tuple[Rational, ...], int]:
    """Return ambient dimension and exact rational coordinates from a V-rep."""

    dimension = len(vertices[0].coordinates)
    points: tuple[tuple[Rational, ...], ...] = tuple(
        tuple(
            Rational(*coordinate.as_integer_ratio())
            for coordinate in vertex.coordinates
        )
        for vertex in vertices
    )
    return points, dimension


def _vertices_from_h_representation(
    halfspaces: tuple[Halfspace, ...],
) -> tuple[list[tuple[Rational, ...]], int]:
    """Enumerate the vertices of an H-representation exactly."""

    dimension = len(halfspaces[0].coefficients)
    reduced = _deduplicate_halfspaces(halfspaces)
    rows = _halfspace_rows(reduced)
    subsystem_count = math.comb(len(rows), dimension)
    if subsystem_count > MAX_SUBSYSTEM_SOLVES:
        raise ValueError(
            "polytope vertex enumeration exceeds the combinatorial bound "
            f"({subsystem_count} > {MAX_SUBSYSTEM_SOLVES} subsystems)"
        )
    return vertices_from_halfspaces(rows, dimension), dimension


def _is_bounded_h(halfspaces: tuple[Halfspace, ...]) -> bool:
    """Decide whether ``{x : A x <= 0}`` contains only the origin."""

    dimension = len(halfspaces[0].coefficients)
    halfspaces = _deduplicate_halfspaces(halfspaces)
    if dimension == 1:
        has_positive = any(
            Rational(*halfspace.coefficients[0].as_integer_ratio()) > 0
            for halfspace in halfspaces
        )
        has_negative = any(
            Rational(*halfspace.coefficients[0].as_integer_ratio()) < 0
            for halfspace in halfspaces
        )
        return has_positive and has_negative
    normals = [
        [
            Rational(*coefficient.as_integer_ratio())
            for coefficient in halfspace.coefficients
        ]
        for halfspace in halfspaces
    ]
    if len(normals) < dimension + 1:
        return False
    try:
        subset_count = math.comb(len(normals), dimension)
    except ValueError:
        subset_count = 10**18
    if subset_count > MAX_BOUNDEDNESS_COMBINATIONS:
        raise ValueError(
            "H-representation boundedness precheck exceeds the "
            f"{MAX_BOUNDEDNESS_COMBINATIONS}-combination budget "
            f"({subset_count} > {MAX_BOUNDEDNESS_COMBINATIONS})"
        )
    return recession_cone_is_trivial(normals, dimension)


def _prepare_volume_components(
    points: Sequence[Sequence[object]],
    dim: int,
) -> tuple[list[list[Any]], list[tuple[int, ...]]]:
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
            return [list(point) for point in unique], []
        _require_interval_volume_within_result_bound(unique)
        return [list(point) for point in unique], []

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
        raise PolytopeAdmissionError(
            "subfacet_bound",
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacets} > {MAX_HULL_SUBFACETS} d-subsets)",
        )
    pts = _filter_redundant_vertices(pts, dim)
    triangulation = _triangulate(pts, dim)
    if len(pts) < dim + 1:
        return pts, []
    if not triangulation:
        return pts, []
    table = [_point_digit_lengths(row) for row in pts]
    _require_triangulated_volume_within_result_bound(table, triangulation, dim)
    return pts, triangulation


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
            raise _validation_error(
                "facet_inequality", "facet inequality must have a nonzero normal"
            )
        entries = [
            Fraction(*value.as_integer_ratio())
            for value in (*self.halfspace.coefficients, self.halfspace.offset)
        ]
        if any(entry.denominator != 1 for entry in entries):
            raise _validation_error(
                "facet_inequality", "facet inequality entries must be integers"
            )
        gcd = 0
        for entry in entries:
            gcd = math.gcd(gcd, abs(int(entry)))
        if gcd != 1:
            raise _validation_error(
                "facet_inequality",
                "facet inequality must be primitive over the integers",
            )
        if any(
            right <= left
            for left, right in zip(
                self.source_vertex_indices,
                self.source_vertex_indices[1:],
                strict=False,
            )
        ):
            raise _validation_error(
                "facet_source_indices",
                "facet source vertex indices must be strictly increasing",
            )
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
    def require_source_bound_profile_shape(self) -> Self:
        if any(len(vertex.coordinates) != self.dimension for vertex in self.vertices):
            raise _validation_error(
                "dimension_bound",
                "every source vertex must have exactly `dimension` coordinates",
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
            raise _validation_error(
                "result_bound",
                "facet profile exceeds the "
                f"{MAX_FACET_INCIDENCES}-incidence result bound",
            )
        if any(
            index >= len(self.vertices)
            for facet in self.facets
            for index in facet.source_vertex_indices
        ):
            raise _validation_error(
                "facet_source_indices",
                "facet source vertex indices must refer to the retained vertices",
            )
        try:
            encode_strict_json(self.model_dump(mode="json"), limits=CanonicalLimits())
        except ValueError as exc:
            raise _validation_error(
                "canonical_json_bound",
                "facet profile exceeds the canonical JSON output bound",
            ) from exc
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        vertices: tuple[Vertex, ...],
        dimension: int,
        facets: tuple[PrimitiveFacet, ...],
    ) -> Self:
        """Build a trusted kernel outcome without replaying its enumeration."""

        return cls.model_construct(
            vertices=vertices,
            dimension=dimension,
            facets=facets,
        )


FacetVertexTuple = Annotated[
    tuple[Vertex, ...],
    Field(min_length=2, max_length=MAX_VERTICES),
]


class FacetIncidenceRequest(StrictModel):
    """A full-dimensional bounded rational V-representation for facet enumeration.

    The representation is given either as bare coordinate vertices or
    unchanged as the domain's canonical labelled ``RationalVPolytope``
    value (for example the ``polytope`` of a support result), constructed
    or serialized.
    """

    vertices: FacetVertexTuple | RationalVPolytope = Field(
        description=(
            "Ordered rational V-representation: bare coordinate vertices or "
            "one canonical labelled ``RationalVPolytope`` value (its "
            "serialized ``space``/``vertices`` shape is accepted too), such "
            "as the ``polytope`` of a support result. The points must "
            "affinely span their "
            "ambient dimension; lower-dimensional hulls are rejected because this "
            "operation returns ambient codimension-one facets. Repeated source rows "
            "are retained for incidence binding but create no candidate hyperplanes "
            "or candidate side tests, so admission requires m*C(m,d) <= "
            f"{MAX_FACET_SIGN_TESTS} candidate-side tests, where m is the number of "
            "distinct rows and every candidate hyperplane spanned by those distinct "
            "rows is side-tested against exactly those distinct rows; the final-facet "
            "incidence scans then range over all n source positions and are bounded "
            "by the materialized-profile result limits below. Both charges apply "
            "during the single owner-local execution. Execution materializes the "
            "complete bounded enumeration, so its exact "
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

    @model_validator(mode="before")
    @classmethod
    def accept_canonical_v_polytope_value(cls, data: object) -> object:
        """Project the canonical labelled V-polytope onto bare vertices.

        Support results carry ``RationalVPolytope`` as the domain's
        canonical V-representation, so composing one into a facet request
        must not force callers to discard the labelled space and rebuild
        every vertex. Both the constructed value and its serialized
        ``space``/``vertices`` shape are accepted unchanged and mapped
        positionally (the labelled axis fixes the coordinate order) before
        ordinary validation, so the operation sees exactly the declared
        V-representation; a serialized value is re-validated as the canonical
        type first. Cheap outer fields and the operation's coordinate envelope
        are preflighted before nested parsing.
        """

        data = canonicalize_json_containers(data)

        if not isinstance(data, dict):
            return data
        value = data.get("vertices")
        carries_v_polytope = isinstance(value, RationalVPolytope) or (
            isinstance(value, dict) and set(value) == {"space", "vertices"}
        )
        if carries_v_polytope:
            unknown_fields = set(data) - {"vertices", "dimension_bound"}
            if unknown_fields:
                raise _validation_error(
                    "facet_incidence_bound",
                    "unexpected fields for a facet incidence request: "
                    f"{sorted(unknown_fields)}",
                )
            axis_count = _v_polytope_axis_count(value)
            if axis_count is not None:
                _require_projected_dimension_bound(
                    axis_count,
                    data.get("dimension_bound", MAX_FACET_DIMENSION),
                    _FACET_DIMENSION_BOUND_ADAPTER,
                    MAX_FACET_DIMENSION,
                )
        if isinstance(value, RationalVPolytope):
            return {**data, "vertices": _canonical_v_polytope_vertices(value)}
        if isinstance(value, dict) and set(value) == {"space", "vertices"}:
            _require_raw_v_polytope_coordinates_within_facet_envelope(value)
            canonical = RationalVPolytope.model_validate(value)
            return {**data, "vertices": _canonical_v_polytope_vertices(canonical)}
        return data


_FACET_DIMENSION_BOUND_ADAPTER: TypeAdapter[int] = TypeAdapter(
    Annotated[
        int,
        *FacetIncidenceRequest.model_fields["dimension_bound"].metadata,
    ],
    config=ConfigDict(strict=True),
)


class RationalCoordinateSpace(StrictModel):
    """One ordered labelled rational coordinate space.

    Coordinate order is mathematical data: a covector component can only
    pair with the point coordinate named by the same position in this axis.
    The axis count reaches ``MAX_FACET_DIMENSION`` so the canonical
    V-representation covers the facet profiles' wider published domain;
    support still pairs spaces with covectors of at most ``MAX_DIMENSION``
    components, and volume rejects over-dimensional values through its own
    dimension bound.
    """

    axes: tuple[CoordinateAxis, ...] = Field(
        min_length=1, max_length=MAX_FACET_DIMENSION
    )

    @model_validator(mode="after")
    def require_distinct_axes(self) -> Self:
        if len(set(self.axes)) != len(self.axes):
            raise _validation_error(
                "coordinate_axes_unique", "coordinate axes must be unique"
            )
        return self


class RationalPolytopeVertex(StrictModel):
    """A labelled exact vertex in a rational coordinate space."""

    vertex_id: Annotated[str, AfterValidator(_require_unicode_scalar_label)] = Field(
        min_length=1,
        max_length=MAX_COORDINATE_LABEL_LENGTH,
    )
    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_FACET_DIMENSION,
    )


class RationalVPolytope(StrictModel):
    """A full-dimensional bounded rational polytope by its exact vertices.

    The vertices are a canonical labelled V-representation: their IDs are
    strictly ordered and their coordinate tuples are distinct.  Whether the
    rows are a full-dimensional irredundant hull is a support-operation
    precondition, proved by its bounded kernel rather than when a canonical
    value is parsed.  This keeps the value neutral for consumers such as
    volume, which intentionally accepts redundant V-representation rows.
    """

    space: RationalCoordinateSpace
    vertices: tuple[RationalPolytopeVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "Ordered distinct V-representation rows. The support operation's "
            "exact extremality proof requires C(n,d) <= "
            f"{MAX_SUPPORT_VERTEX_SUBSETS} candidate subfacets, C(n,d) * "
            f"(n-d) <= {MAX_SUPPORT_ORIENTATION_TESTS} orientation tests, and "
            f"C(n,d) * (n-d) * D^2 <= {MAX_EXTREMALITY_HEIGHT_WORK}, where D is "
            "the largest reduced numerator/denominator digit count across all "
            "vertex coordinates (exact determinant work grows quadratically "
            "with coordinate height)."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_full_dimensional_vertices(self) -> Self:
        dimension = len(self.space.axes)
        if len(self.vertices) < dimension + 1:
            raise _validation_error(
                "dimension_bound",
                "a full-dimensional V-polytope needs at least dimension + 1 vertices",
            )
        vertex_ids = tuple(vertex.vertex_id for vertex in self.vertices)
        if tuple(sorted(vertex_ids)) != vertex_ids or len(set(vertex_ids)) != len(
            vertex_ids
        ):
            raise _validation_error(
                "vertex_ids", "vertex IDs must be unique and strictly ordered"
            )
        coordinates = tuple(vertex.coordinates for vertex in self.vertices)
        if any(len(point) != dimension for point in coordinates):
            raise _validation_error(
                "polytope_vertices",
                "every vertex must use the polytope coordinate axis",
            )
        if len(set(coordinates)) != len(coordinates):
            raise _validation_error(
                "polytope_vertices", "polytope vertices must have distinct coordinates"
            )
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
            raise _validation_error(
                "covector_components",
                "covector components must use the declared coordinate axis",
            )
        return self


_FACE_WIRE_RESERVE_BYTES = 65_536
"""Fixed encoded-size allowance for a face's labels and JSON structure.

Axis labels and vertex IDs are short Unicode scalar strings whose RFC 8785
escaping expands each code point to at most twelve characters, so the
combined worst case across the declared container maxima — plus every
structural token of the encoded face — stays inside this reserve.
"""

_FACE_VERTEX_WIRE_OVERHEAD_BYTES = 256
"""Encoded-size allowance for one vertex's structural JSON tokens."""

_FACE_COORDINATE_WIRE_OVERHEAD_BYTES = 48
"""Encoded-size allowance beyond its two canonical integer strings for one
serialized rational coordinate."""


def _face_coordinate_wire_bytes(component: object) -> int | None:
    """Return one authored coordinate's conservative serialized height.

    The reduced numerator/denominator strings are measured exactly as the
    strict JSON encoding writes them; unrecognized shapes return ``None``
    so ordinary nested validation reports them with the published schema
    errors.
    """

    if isinstance(component, CanonicalRational):
        num, den = component.num, component.den
    elif isinstance(component, dict):
        raw_num = component.get("num")
        raw_den = component.get("den")
        if not isinstance(raw_num, str) or not isinstance(raw_den, str):
            return None
        num, den = raw_num, raw_den
    else:
        return None
    return len(num) + len(den) + _FACE_COORDINATE_WIRE_OVERHEAD_BYTES


def _estimate_face_wire_bytes(data: object) -> int:
    """Conservatively upper-bound one authored exposed face's encoded size.

    Every recognized coordinate contributes the length of both canonical
    integer strings plus per-element overhead; unrecognized shapes
    contribute nothing here and remain the business of ordinary nested
    validation and the exact strict JSON replay. The estimate may only
    over-count, so an accepted aggregate always encodes at or under it.
    """

    total = _FACE_WIRE_RESERVE_BYTES
    for vertex in _iter_raw_entries(data, "vertices"):
        total += _FACE_VERTEX_WIRE_OVERHEAD_BYTES
        for component in _iter_raw_entries(vertex, "coordinates"):
            coordinate_bytes = _face_coordinate_wire_bytes(component)
            if coordinate_bytes is not None:
                total += coordinate_bytes
    return total


class RationalExposedFace(StrictModel):
    """The complete vertex family of one exposed face of a V-polytope.

    The aggregate encoded payload must fit the domain's strict JSON
    transport limit: an accepted face composes unchanged across the
    supported serialization boundary, so a face whose coordinates alone
    exceed ``CanonicalLimits().max_output_bytes`` is rejected here as a
    typed validation error.
    """

    space: RationalCoordinateSpace
    vertices: tuple[RationalPolytopeVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
    )

    @model_validator(mode="before")
    @classmethod
    def require_aggregate_wire_size_within_transport_bound(cls, data: object) -> object:
        """Reject faces whose aggregate payload overflows the transport limit.

        Declared container maxima admit faces whose canonical coordinates
        alone encode far past ``CanonicalLimits().max_output_bytes``, yet a
        value only composes through boundaries ``encode_strict_json``
        supports. Nested canonical-rational parsing validates every authored
        coordinate before parent after-validators run, so gating there would
        pay the complete parse of a guaranteed-to-fail payload. This gate
        conservatively estimates the encoded height from the authored
        reduced-component strings — dict or built value alike — and rejects
        an over-limit aggregate before any coordinate is parsed; the
        residual gap between the estimate and the exact encoded size stays
        covered by the strict JSON replay in
        ``require_canonical_face_vertices``.
        """

        data = canonicalize_json_containers(data)

        estimated = _estimate_face_wire_bytes(data)
        if estimated > CanonicalLimits().max_output_bytes:
            raise _validation_error(
                "canonical_json_bound",
                "exposed face exceeds the canonical JSON output bound",
            )
        return data

    @model_validator(mode="after")
    def require_canonical_face_vertices(self) -> Self:
        vertex_ids = tuple(vertex.vertex_id for vertex in self.vertices)
        if tuple(sorted(vertex_ids)) != vertex_ids or len(set(vertex_ids)) != len(
            vertex_ids
        ):
            raise _validation_error(
                "exposed_face_vertex",
                "exposed-face vertex IDs must be unique and strictly ordered",
            )
        dimension = len(self.space.axes)
        if any(len(vertex.coordinates) != dimension for vertex in self.vertices):
            raise _validation_error(
                "exposed_face_vertex",
                "every exposed-face vertex must use the face coordinate axis",
            )
        if len({vertex.coordinates for vertex in self.vertices}) != len(self.vertices):
            raise _validation_error(
                "exposed_face_coordinates_unique",
                "exposed-face vertices must have distinct coordinates",
            )
        try:
            encode_strict_json(self.model_dump(mode="json"), limits=CanonicalLimits())
        except ValueError as exc:
            raise _validation_error(
                "canonical_json_bound",
                "exposed face exceeds the canonical JSON output bound",
            ) from exc
        return self


def require_support_components_within_envelope(
    polytope: RationalVPolytope,
    covector: RationalCovector,
) -> None:
    """Enforce the support operation's per-component execution envelope.

    Canonical polytope values admit every canonical rational coordinate;
    this smaller operation-specific bound is the single admission decision
    enforced by the native ``polytope_support`` entry point, keeping the exact
    hull intermediates of one accepted call
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
    a canonical rational within the published per-component digit envelope,
    which is stricter than the global canonical limit.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute one support value and its complete exposed vertex face. "
                "The full-dimensional exact V-representation and covector use "
                "one identical labelled coordinate space. Each reduced numerator "
                "and denominator is limited to "
                f"{MAX_SUPPORT_COMPONENT_DIGITS} digits before evaluation."
            )
        }
    )

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
        components sit between this operation's component envelope and the
        global canonical limit would reach the exact hull-facet proof
        inside ``RationalVPolytope`` construction before
        native operation could reject it. This
        preflight measures only the authored reduced components of the raw
        payload — dict or already-built values alike — so even a rejected
        request stays inside the advertised execution envelope; the
        canonical V-polytope value's broader domain is unchanged. The
        request's closed outer field set is preflighted first for the same
        reason: forbidden extras are reported by ``StrictModel`` only
        after every declared field has been parsed.
        """

        data = canonicalize_json_containers(data)

        canonical = _preflight_raw_support_components(data)
        _require_raw_support_request_shape(canonical)
        return canonical

    @model_validator(mode="after")
    def require_common_coordinate_space(self) -> Self:
        if self.polytope.space != self.covector.space:
            raise _validation_error(
                "coordinate_space_mismatch",
                "polytope and covector must use the same coordinate space",
            )
        return self


class PolytopeSupportResult(StrictModel):
    """A source-bound exact support value and its complete exposed face.

    The retained source satisfies the same admitted execution envelope as
    ``PolytopeSupportRequest``: each polytope vertex coordinate and covector
    component stays within the published support-component bound.
    """

    polytope: RationalVPolytope
    covector: RationalCovector
    support_value: CanonicalRational
    exposed_face: RationalExposedFace

    @model_validator(mode="before")
    @classmethod
    def require_raw_components_within_support_envelope(cls, data: object) -> object:
        """Preflight the retained source before nested V-polytope parsing.

        Nested values are constructed before parent after-validators run, so
        the same raw-payload measurement as the request rejects an
        over-envelope retained source early. The result's outer shape and
        conclusion fields are preflighted at the same boundary.
        """

        data = canonicalize_json_containers(data)

        canonical = _preflight_raw_support_components(data)
        _require_raw_support_conclusions_admissible(canonical)
        return canonical

    @model_validator(mode="after")
    def require_source_and_conclusion_shape(self) -> Self:
        if self.polytope.space != self.covector.space:
            raise _validation_error(
                "coordinate_space_mismatch",
                "polytope and covector must use the same coordinate space",
            )
        if self.exposed_face.space != self.polytope.space:
            raise _validation_error(
                "coordinate_space_mismatch",
                "exposed face must use the same coordinate space as the polytope",
            )
        source_by_id = {vertex.vertex_id: vertex for vertex in self.polytope.vertices}
        if any(
            source_by_id.get(vertex.vertex_id) != vertex
            for vertex in self.exposed_face.vertices
        ):
            raise _validation_error(
                "exposed_face_source",
                "exposed-face vertices must occur unchanged in the retained polytope",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polytope: RationalVPolytope,
        covector: RationalCovector,
        support_value: CanonicalRational,
        exposed_face: RationalExposedFace,
    ) -> Self:
        """Build one trusted kernel outcome without replaying its dot products."""

        return cls.model_construct(
            polytope=polytope,
            covector=covector,
            support_value=support_value,
            exposed_face=exposed_face,
        )


def _canonical_v_polytope_vertices(polytope: RationalVPolytope) -> tuple[Vertex, ...]:
    """Map the labelled canonical V-polytope onto bare vertices.

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


def _require_projected_dimension_bound(
    dimension: int,
    dimension_bound: object,
    bound_adapter: TypeAdapter[int],
    upper_bound: int,
) -> None:
    """Reject a V-polytope outside the consumer's published dimension bound.

    The raw bound is measured with the consumer's own ``dimension_bound``
    field schema, derived from its declaration so the constraint range
    cannot drift, under the strict validation boundary every ``math.run``
    request passes through: an integer within ``[1, upper_bound]`` bounds
    the comparison exactly as the outer model would, while strings,
    floats, booleans, null, and out-of-range values are rejected here before
    nested canonical parsing.
    """

    if dimension_bound is None:
        raise _validation_error(
            "dimension_bound",
            f"dimension_bound must be an integer between 1 and {upper_bound}",
        )
    try:
        bound: int = bound_adapter.validate_python(dimension_bound)
    except ValidationError as exc:
        raise _validation_error(
            "dimension_bound",
            f"dimension_bound must be an integer between 1 and {upper_bound}",
        ) from exc
    if dimension > bound:
        raise _validation_error(
            "dimension_bound",
            f"dimension {dimension} exceeds the dimension bound {bound}",
        )


VertexTuple = Annotated[
    tuple[Vertex, ...],
    Field(min_length=1, max_length=MAX_VERTICES),
]


class PolytopeVolumeRequest(StrictModel):
    """A bounded rational polytope in exactly one of the two representations.

    The V-representation is given either as bare vertices or unchanged as
    the domain's canonical labelled ``RationalVPolytope`` value (for
    example the ``polytope`` of a support result), constructed or
    serialized; admission enforces the same work bound on both forms.

    Admission enforces a work bound that couples vertex count with ambient
    dimension: after duplicate points are removed, the exact hull
    enumeration considers ``C(n, d)`` d-subsets of ``n`` distinct vertices.
    The same named hull-work limit applies to a vertex set derived from an
    H-representation, whose distinct rows have their own named boundedness
    limit. The field descriptions publish both exact rules.
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
            f"rejected. Within the {MAX_VERTICES}-vertex maximum this admits "
            f"up to {_largest_combination_axis(ceiling=MAX_VERTICES, dimension=3, work=MAX_HULL_SUBFACETS)} "
            "distinct vertices for d <= 3, "
            f"{_largest_combination_axis(ceiling=MAX_VERTICES, dimension=4, work=MAX_HULL_SUBFACETS)} for d = 4, "
            f"{_largest_combination_axis(ceiling=MAX_VERTICES, dimension=5, work=MAX_HULL_SUBFACETS)} for d = 5, and "
            f"{_largest_combination_axis(ceiling=MAX_VERTICES, dimension=6, work=MAX_HULL_SUBFACETS)} for d = 6."
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
            f"normals); within the {MAX_FACETS}-row maximum this admits "
            f"{_largest_combination_axis(ceiling=MAX_FACETS, dimension=4, work=MAX_BOUNDEDNESS_COMBINATIONS)} distinct "
            "half-spaces for d <= 4, "
            f"{_largest_combination_axis(ceiling=MAX_FACETS, dimension=5, work=MAX_BOUNDEDNESS_COMBINATIONS)} for d = 5, and "
            f"{_largest_combination_axis(ceiling=MAX_FACETS, dimension=6, work=MAX_BOUNDEDNESS_COMBINATIONS)} for d = 6. The "
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
        ordinary validation, so the operation sees exactly the declared
        V-representation; a serialized value is re-validated as the canonical
        type first. The closed field set, halfspace conflict, and published
        ``dimension_bound`` schema are preflighted before nested parsing.
        """

        data = canonicalize_json_containers(data)

        if not isinstance(data, dict):
            return data
        value = data.get("vertices")
        carries_v_polytope = isinstance(value, RationalVPolytope) or (
            isinstance(value, dict) and set(value) == {"space", "vertices"}
        )
        if carries_v_polytope:
            unknown_fields = set(data) - {"vertices", "halfspaces", "dimension_bound"}
            if unknown_fields:
                raise _validation_error(
                    "halfspaces",
                    "unexpected fields for a polytope volume request: "
                    f"{sorted(unknown_fields)}",
                )
            if data.get("halfspaces") is not None:
                raise _validation_error(
                    "halfspaces",
                    "exactly one of `vertices` or `halfspaces` must be provided",
                )
            axis_count = _v_polytope_axis_count(value)
            if axis_count is not None:
                _require_projected_dimension_bound(
                    axis_count,
                    data.get("dimension_bound", MAX_DIMENSION),
                    _DIMENSION_BOUND_ADAPTER,
                    MAX_DIMENSION,
                )
        if isinstance(value, RationalVPolytope):
            return {**data, "vertices": _canonical_v_polytope_vertices(value)}
        if isinstance(value, dict) and set(value) == {"space", "vertices"}:
            canonical = RationalVPolytope.model_validate(value)
            return {**data, "vertices": _canonical_v_polytope_vertices(canonical)}
        return data

    @model_validator(mode="after")
    def validate_representation(self) -> Self:
        has_v = self.vertices is not None
        has_h = self.halfspaces is not None
        if has_v == has_h:
            raise _validation_error(
                "halfspaces",
                "exactly one of `vertices` or `halfspaces` must be provided",
            )
        return self


_DIMENSION_BOUND_ADAPTER: TypeAdapter[int] = TypeAdapter(
    Annotated[int, *PolytopeVolumeRequest.model_fields["dimension_bound"].metadata],
    config=ConfigDict(strict=True),
)


def _validate_vertices(
    vertices: tuple[Vertex, ...], dimension_bound: int
) -> tuple[list[list[Any]], int, list[tuple[int, ...]]]:
    """Validate a V-representation: count, per-component, and dimension bounds."""
    if len(vertices) < 1:
        raise _validation_error("vertices_bound", "`vertices` must be non-empty")
    if len(vertices) > MAX_VERTICES:
        raise _validation_error(
            "vertices_bound", f"`vertices` exceeds the {MAX_VERTICES}-vertex bound"
        )
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
        raise _validation_error(
            "dimension_bound",
            f"dimension {dim} exceeds the dimension bound {dimension_bound}",
        )
    for vertex in vertices:
        if len(vertex.coordinates) != dim:
            raise _validation_error(
                "vertex_dimension_consistency", "all vertices must share one dimension"
            )

    # Exact-volume growth is bounded over the whole triangulation, so the
    # same admission runs on the rational points themselves; it applies
    # the combinatorial hull-work bound after exact deduplication,
    # mirroring the kernel pipeline.
    points, resolved_dim = _vertices_from_v_representation(vertices)
    prepared, triangulation = _prepare_volume_components(points, resolved_dim)
    return prepared, resolved_dim, triangulation


def _require_admissible_h_vertices(
    halfspaces: tuple[Halfspace, ...], dim: int
) -> tuple[list[list[Any]], list[tuple[int, ...]]]:
    """Admit the derived vertex set of an H-representation.

    Bounded-ness and non-emptiness must be decided before any exact
    enumeration: an unbounded or empty H-polytope is not a valid request,
    so it is rejected here as ``ValidationError`` rather than as a host
    exception after acceptance.  The derived vertices then drive the same
    brute-force hull enumeration and exact-volume growth bound as a
    caller-supplied V-representation, so the identical combinatorial and
    result-size admission applies before accepting the request.
    """

    if not _is_bounded_h(halfspaces):
        raise _validation_error(
            "halfspaces",
            "the H-representation is unbounded; polytope volume requires a bounded polytope",
        )
    verts, _resolved_dim = _vertices_from_h_representation(halfspaces)
    if not verts:
        raise _validation_error(
            "h_representation", "the H-representation defines an empty polytope"
        )
    subfacets = math.comb(len(verts), dim)
    if subfacets > MAX_HULL_SUBFACETS:
        raise _validation_error(
            "halfspace_coefficients",
            "polytope hull enumeration exceeds the combinatorial bound "
            f"({subfacets} > {MAX_HULL_SUBFACETS} d-subsets)",
        )
    # Solved vertices can carry more digits than the declaring half-space
    # coefficients, so measure them directly.
    return _prepare_volume_components(verts, dim)


def _validate_halfspaces(
    halfspaces: tuple[Halfspace, ...], dimension_bound: int
) -> tuple[list[list[Any]], int, list[tuple[int, ...]]]:
    """Validate an H-representation: count, per-component, and dimension bounds."""
    if len(halfspaces) < 1:
        raise _validation_error("halfspaces", "`halfspaces` must be non-empty")
    if len(halfspaces) > MAX_FACETS:
        raise _validation_error(
            "halfspaces", f"`halfspaces` exceeds the {MAX_FACETS}-facet bound"
        )
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
        raise _validation_error(
            "halfspaces",
            f"dimension {dim} exceeds the dimension bound {dimension_bound}",
        )
    for halfspace in halfspaces:
        if len(halfspace.coefficients) != dim:
            raise _validation_error(
                "halfspaces", "all half-spaces must share one dimension"
            )
    for halfspace in halfspaces:
        if all(c.as_fraction() == 0 for c in halfspace.coefficients):
            raise _validation_error(
                "halfspaces", "half-space coefficients must not all be zero"
            )
    prepared, triangulation = _require_admissible_h_vertices(halfspaces, dim)
    return prepared, dim, triangulation


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
    "MAX_EXTREMALITY_HEIGHT_WORK",
    "MAX_FACETS",
    "MAX_FACET_COORDINATE_DIGITS",
    "MAX_FACET_DIMENSION",
    "MAX_FACET_INCIDENCES",
    "MAX_FACET_SIGN_TESTS",
    "MAX_HULL_SUBFACETS",
    "MAX_SUPPORT_COMPONENT_DIGITS",
    "MAX_SUPPORT_ORIENTATION_TESTS",
    "MAX_SUPPORT_VERTEX_SUBSETS",
    "MAX_VERTICES",
    "FacetIncidenceRequest",
    "FacetIncidenceResult",
    "Halfspace",
    "PolytopeSupportRequest",
    "PolytopeSupportResult",
    "PolytopeVolumeRequest",
    "PolytopeVolumeResult",
    "PrimitiveFacet",
    "RationalCoordinateSpace",
    "RationalCovector",
    "RationalExposedFace",
    "RationalPolytopeVertex",
    "RationalVPolytope",
    "Vertex",
]
