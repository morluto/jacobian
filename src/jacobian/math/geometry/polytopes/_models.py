"""Request/result models for the bounded rational polytope domain."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from fractions import Fraction
from itertools import combinations
from typing import Annotated, Any, Self, cast

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic_core import PydanticCustomError
from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    MAX_DD_COEFFICIENT_DIGITS,
    MAX_DD_PAIR_BOUND,
    MAX_DD_RAY_BOUND,
    MAX_DD_WEIGHTED_HEIGHT_WORK,
    MAX_PULLING_SIMPLEX_BOUND,
    halfspaces_to_generators,
    maximal_incidence_faces,
    points_to_facets,
    pulling_triangulation,
    rational_rank,
    require_pulling_work_admissible,
)
from jacobian.math.geometry.polytopes._rational_geometry import (
    determinant_sign,
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

    # Pydantic's stubs restrict these to literals for localization safety, but
    # this domain intentionally constructs stable owner-prefixed error codes.
    return PydanticCustomError(
        f"polytope.{reason}",  # pyright: ignore[reportArgumentType]
        message,  # pyright: ignore[reportArgumentType]
    )


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
differences. This conservative height bounds those private intermediates and
the digits in the primitive integer facet rows.
"""

MAX_COMPUTED_FACETS = 256
"""Maximum number of canonical facets materialized by one result."""

MAX_FACET_INCIDENCES = 16_384
"""Maximum total source-row/facet incidences materialized by one result."""

MAX_VERTICES = 64
"""Absolute upper bound on the number of vertices in a V-representation.

The private conversion kernel applies output-sensitive ray, candidate-pair,
and coefficient-growth admission after exact deduplication. Structured box
and simplex hulls have tighter direct bounds.
"""

MAX_POLYTOPE_FACE_LATTICE_DIMENSION = 3
MAX_POLYTOPE_FACE_LATTICE_FACETS = 2 * MAX_VERTICES - 4
MAX_POLYTOPE_FACE_LATTICE_EDGES = 3 * MAX_VERTICES - 6
MAX_POLYTOPE_FACE_LATTICE_FACES = 6 * MAX_VERTICES - 8
MAX_POLYTOPE_FACE_LATTICE_COVERS = 15 * MAX_VERTICES - 28
MAX_POLYTOPE_FACE_LATTICE_WORK = 1_000_000
MAX_POLYTOPE_FACE_LATTICE_RESULT_CHARS = 2_000_000

MAX_FACETS = 64
"""Absolute upper bound on the number of half-spaces in an H-representation."""

MAX_COORDINATE_LABEL_LENGTH = 64
"""Maximum Unicode-scalar length of an axis or vertex identifier."""

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
inputs and the exact construction work required by the public
V-polytope value.
"""

MAX_EXTREMALITY_HEIGHT_WORK = 20_000_000_000
"""Ceiling coupling extremality-proof work with coordinate height.

The DD upper bound supplies ``T`` candidate pairs and exact rank checks before
conversion. With ``D`` the largest reduced component digit count, admission
charges ``T * (d + 1) * D^2``: every pair combines ``d + 1`` coordinates and
integer multiplication is at least quadratic under this conservative model.
Structured hulls still pass this cheap count-height gate before their tighter
box or simplex presolve runs.
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
        require_bounded_rational(component, max_digits=max_digits, label=label)
        return
    elif isinstance(component, dict):
        raw_num = component.get("num")
        raw_den = component.get("den")
        for value in (raw_num, raw_den):
            if (isinstance(value, str) and len(value.lstrip("-")) > max_digits) or (
                type(value) is int and abs(value) >= 10**max_digits
            ):
                raise _validation_error(
                    "component_digit_bound",
                    f"{label} exceeds the {max_digits}-digit bound",
                )
    else:
        return


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
    if isinstance(component, dict) and set(component) == {"num", "den"}:
        try:
            if isinstance(component["num"], str) and isinstance(component["den"], str):
                CanonicalRational.model_validate_json(json.dumps(component))
            else:
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
            (
                format_canonical_integer(component.num),
                format_canonical_integer(component.den),
            )
            for component in vertex.coordinates
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
        (
            format_canonical_integer(component.num),
            format_canonical_integer(component.den),
        )
        if isinstance(component, CanonicalRational)
        else (
            format_canonical_integer(component["num"])
            if type(component["num"]) is int
            else component["num"],
            format_canonical_integer(component["den"])
            if type(component["den"]) is int
            else component["den"],
        )
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
    """Bound the exact determinant sum using one global axis denominator.

    For axis ``j``, the product of all authored coordinate denominators is a
    common denominator ``L_j`` for every simplex entry on that axis.  Hence
    every determinant shares ``prod(L_j)``: summing simplices adds only the
    decimal length of the simplex count, not the product of every simplex's
    denominators.  This remains conservative while avoiding the old false
    exponential growth on integral cubes and other highly triangulated hulls.
    """

    axis_denominator_digits = [
        sum(row[axis][1] for row in table) for axis in range(dim)
    ]
    determinant_numerator_digits = sum(
        max(row[axis][0] for row in table) + axis_denominator_digits[axis] + 2
        for axis in range(dim)
    )
    factorial_digits = len(str(math.factorial(dim)))
    carry = len(str(len(triangulation))) + dim + 4
    numerator_total = determinant_numerator_digits + carry
    denominator_total = sum(axis_denominator_digits) + factorial_digits + carry
    if (
        numerator_total > MAX_RESULT_COMPONENT_DIGITS
        or denominator_total > MAX_RESULT_COMPONENT_DIGITS
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
            sign = determinant_sign(
                [[points[i][k] for k in range(dim)] + [1] for i in subset]
                + [[points[p][k] for k in range(dim)] + [1]]
            )
            if sign > 0:
                signs.add(1)
            elif sign < 0:
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
    vec = [cast(Rational, Rational(nullspace[0][j])) for j in range(dim + 1)]
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
    active_normals: list[list[Sequence[object]]] = [[] for _ in range(point_count)]
    for normal, members in groups.items():
        normal_values = list(normal[:-1])
        for index in members:
            if 0 <= index < point_count:
                counts[index] += 1
                active_normals[index].append(normal_values)
    kept = [
        index
        for index in range(point_count)
        if active_normals[index] and rational_rank(active_normals[index], dim) == dim
    ]
    return kept, counts


def _filter_redundant_vertices(
    points: list[list[Rational]], dim: int
) -> list[list[Rational]]:
    """Return extreme hull vertices, dropping redundant boundary points."""

    if len(points) <= dim:
        return points
    hull = points_to_facets(points, dim)
    active_normals: list[list[Sequence[object]]] = [[] for _ in points]
    for (normal, _offset), incidence in zip(
        hull.facets, hull.facet_incidence, strict=True
    ):
        for index in range(len(points)):
            if incidence & (1 << index):
                active_normals[index].append(list(normal))
    keep_indices = [
        index
        for index, normals in enumerate(active_normals)
        if normals and rational_rank(normals, dim) == dim
    ]
    if len(keep_indices) < dim + 1:
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


def _rank_of_diffs(points: Sequence[Sequence[Any]], dim: int) -> int:
    """Rank of the matrix of ``point - point[0]`` differences in ``dim`` dims."""

    if len(points) <= 1:
        return 0
    reference = points[0]
    differences = [
        [points[index][axis] - reference[axis] for axis in range(dim)]
        for index in range(1, len(points))
    ]
    return rational_rank(differences, dim)


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
        for fraction in (*fractions, offset):
            lcm = lcm * fraction.denominator // math.gcd(lcm, fraction.denominator)
        ints = [int(fraction * lcm) for fraction in fractions]
        rhs = int(offset * lcm)
        gcd = 0
        for integer in (*ints, rhs):
            gcd = math.gcd(gcd, abs(integer))
        normalized = (
            tuple(integer // gcd for integer in ints),
            (rhs // gcd, 1),
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
                cast(Rational, Rational(*coefficient.as_integer_ratio()))
                for coefficient in hs.coefficients
            ],
            cast(Rational, Rational(*hs.offset.as_integer_ratio())),
        )
        for hs in halfspaces
    ]


def _vertices_from_v_representation(
    vertices: tuple[Vertex, ...],
) -> tuple[tuple[tuple[Rational, ...], ...], int]:
    """Return ambient dimension and exact rational coordinates from a V-rep."""

    dimension = len(vertices[0].coordinates)
    points = tuple(
        tuple(
            cast(Rational, Rational(*coordinate.as_integer_ratio()))
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
    return vertices_from_halfspaces(rows, dimension), dimension


def _is_bounded_h(halfspaces: tuple[Halfspace, ...]) -> bool:
    """Decide whether ``{x : A x <= 0}`` contains only the origin."""

    dimension = len(halfspaces[0].coefficients)
    halfspaces = _deduplicate_halfspaces(halfspaces)
    normals = [
        [
            cast(Rational, Rational(*coefficient.as_integer_ratio()))
            for coefficient in halfspace.coefficients
        ]
        for halfspace in halfspaces
    ]
    return recession_cone_is_trivial(normals, dimension)


def _prepare_volume_components(
    points: Sequence[Sequence[object]],
    dim: int,
) -> tuple[list[list[Any]], list[tuple[int, ...]]]:
    """Reject inputs whose exact summed volume cannot fit the canonical type.

    The kernel sums simplex determinants over a whole triangulation, so
    admission must account for denominators contributed by *all* simplices,
    not only ``dim + 1`` vertices.  The guard mirrors the execution
    pipeline — exact deduplication, output-sensitive conversion, extremal
    filtering, and incidence-driven pulling — so an empty triangulation means
    exact volume zero. Repeated points neither inflate DD admission nor let an
    unrepresentable result skip the retained triangulation growth bound.
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

    # Deduplicate before conversion so repeated generators cannot inflate
    # incidence work or survive into the retained triangulation.
    pts = [list(point) for point in _deduplicate_exact_points(points)]
    if len(pts) < dim + 1:
        return pts, []
    if _rank_of_diffs(pts, dim) < dim:
        return pts, []

    hull = points_to_facets(pts, dim)
    extreme_indices: list[int] = []
    for point_index in range(len(pts)):
        active_normals = [
            list(normal)
            for (normal, _offset), incidence in zip(
                hull.facets, hull.facet_incidence, strict=True
            )
            if incidence & (1 << point_index)
        ]
        if active_normals and rational_rank(active_normals, dim) == dim:
            extreme_indices.append(point_index)
    if len(extreme_indices) < dim + 1:
        return pts, []

    if len(extreme_indices) != len(pts):
        remapped_incidence: list[int] = []
        for incidence in hull.facet_incidence:
            bits = 0
            for new_index, old_index in enumerate(extreme_indices):
                if incidence & (1 << old_index):
                    bits |= 1 << new_index
            remapped_incidence.append(bits)
        pts = [pts[index] for index in extreme_indices]
    else:
        remapped_incidence = list(hull.facet_incidence)
    require_pulling_work_admissible(len(remapped_incidence), dim)
    triangulation = list(pulling_triangulation(len(pts), dim, remapped_incidence))
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
        if any(index < 0 for index in self.source_vertex_indices):
            raise _validation_error(
                "facet_source_indices", "facet indices must be nonnegative"
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
        if any(
            len(facet.halfspace.coefficients) != self.dimension for facet in self.facets
        ):
            raise _validation_error(
                "facet_dimension",
                "facet normals must use the retained source coordinate axis",
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
            "are retained for incidence binding but collapse to one homogeneous "
            "generator for conversion. Admission uses the output-sensitive "
            f"double-description ceilings of {MAX_DD_RAY_BOUND} rays, "
            f"{MAX_DD_PAIR_BOUND} candidate pairs, and "
            f"{MAX_DD_COEFFICIENT_DIGITS} primitive-minor digits. The exact "
            "incidence profile is proven to fit the "
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
    def accept_canonical_v_polytope_value(
        cls, data: object, info: ValidationInfo
    ) -> object:
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
            canonical = (
                RationalVPolytope.model_validate_json(json.dumps(value))
                if info.mode == "json"
                else RationalVPolytope.model_validate(value)
            )
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
            "exact extremality proof uses the shared output-sensitive DD work "
            f"bounds ({MAX_DD_RAY_BOUND} rays and {MAX_DD_PAIR_BOUND} pairs) "
            "and a dimension-weighted coefficient-height budget of "
            f"{MAX_EXTREMALITY_HEIGHT_WORK}."
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


class PolytopeFace(StrictModel):
    """One face of a three-dimensional polytope, including bottom and top."""

    dimension: StrictInt = Field(ge=-1, le=MAX_POLYTOPE_FACE_LATTICE_DIMENSION)
    source_vertex_indices: tuple[StrictInt, ...] = Field(max_length=MAX_VERTICES)

    @model_validator(mode="after")
    def require_canonical_face_label(self) -> Self:
        if tuple(sorted(set(self.source_vertex_indices))) != self.source_vertex_indices:
            raise _validation_error(
                "face_lattice_vertex_indices",
                "face vertex indices must be strictly increasing",
            )
        expected_sizes = {-1: 0, 0: 1, 1: 2}
        if self.dimension in expected_sizes:
            expected_size = expected_sizes[self.dimension]
            if len(self.source_vertex_indices) != expected_size:
                raise _validation_error(
                    "face_lattice_face_dimension",
                    "empty, vertex, and edge face labels have invalid sizes",
                )
        elif len(self.source_vertex_indices) < self.dimension + 1:
            raise _validation_error(
                "face_lattice_face_dimension",
                "a face must contain at least dimension + 1 vertices",
            )
        return self


class PolytopeFaceCover(StrictModel):
    """One Hasse cover, indexed into a canonical face tuple."""

    lower_face_index: StrictInt = Field(ge=0, le=MAX_POLYTOPE_FACE_LATTICE_FACES - 1)
    upper_face_index: StrictInt = Field(ge=0, le=MAX_POLYTOPE_FACE_LATTICE_FACES - 1)


class PolytopeFaceLatticeRequest(StrictModel):
    """A labelled exact V-representation whose hull is a 3-polytope."""

    polytope: RationalVPolytope = Field(
        description=(
            "Full-dimensional rational V-representation in three dimensions. "
            "The operation proves its complete facet incidence internally, "
            "then returns the complete face lattice and cover relations."
        )
    )


def _complete_face_lattice_labels(
    extreme_vertex_indices: tuple[int, ...],
    facet_labels: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], tuple[tuple[int, int], ...]]:
    """Reconstruct rank-three face labels from bounded facet incidences."""

    if len(facet_labels) > MAX_POLYTOPE_FACE_LATTICE_FACETS:
        raise _validation_error(
            "face_lattice_facets",
            "the stored facet count exceeds the rank-three planar graph bound",
        )
    edge_occurrences: dict[tuple[int, int], int] = {}
    for left, right in combinations(facet_labels, 2):
        common = tuple(sorted(set(left).intersection(right)))
        if len(common) == 2:
            edge_occurrences[common] = edge_occurrences.get(common, 0) + 1
    if any(count != 1 for count in edge_occurrences.values()):
        raise _validation_error(
            "face_lattice_edges",
            "every edge must be the intersection of exactly two facets",
        )
    if len(edge_occurrences) != len(extreme_vertex_indices) + len(facet_labels) - 2:
        raise _validation_error(
            "face_lattice_euler_identity",
            "the stored facets and edges must satisfy the rank-three Euler identity",
        )
    if len(edge_occurrences) > MAX_POLYTOPE_FACE_LATTICE_EDGES:
        raise _validation_error(
            "face_lattice_edges",
            "the stored edge count exceeds the rank-three planar graph bound",
        )
    face_keys = tuple(
        sorted(
            (
                (-1, ()),
                *((0, (index,)) for index in extreme_vertex_indices),
                *((1, edge) for edge in edge_occurrences),
                *((2, facet) for facet in facet_labels),
                (3, extreme_vertex_indices),
            )
        )
    )
    return face_keys, tuple(edge_occurrences)


def _complete_face_lattice_covers(
    face_keys: tuple[tuple[int, tuple[int, ...]], ...],
    extreme_vertex_indices: tuple[int, ...],
    facet_labels: tuple[tuple[int, ...], ...],
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Reconstruct every rank-adjacent incidence without geometric replay."""

    face_indices = {key: index for index, key in enumerate(face_keys)}
    bottom_index = face_indices[(-1, ())]
    top_index = face_indices[(3, extreme_vertex_indices)]
    vertex_indices = {
        vertices[0]: index
        for (dimension, vertices), index in face_indices.items()
        if dimension == 0
    }
    cover_pairs: set[tuple[int, int]] = {
        (bottom_index, vertex_indices[vertex]) for vertex in extreme_vertex_indices
    }
    for edge in edges:
        edge_index = face_indices[(1, edge)]
        cover_pairs.update((vertex_indices[vertex], edge_index) for vertex in edge)
        cover_pairs.update(
            (edge_index, face_indices[(2, facet)])
            for facet in facet_labels
            if edge[0] in facet and edge[1] in facet
        )
    cover_pairs.update((face_indices[(2, facet)], top_index) for facet in facet_labels)
    return tuple(sorted(cover_pairs))


class PolytopeFaceLatticeResult(StrictModel):
    """Complete source-bound face lattice of the convex hull of ``polytope``."""

    polytope: RationalVPolytope
    extreme_vertex_indices: tuple[StrictInt, ...] = Field(
        min_length=4, max_length=MAX_VERTICES
    )
    faces: tuple[PolytopeFace, ...] = Field(
        min_length=16, max_length=MAX_POLYTOPE_FACE_LATTICE_FACES
    )
    covers: tuple[PolytopeFaceCover, ...] = Field(
        min_length=32, max_length=MAX_POLYTOPE_FACE_LATTICE_COVERS
    )

    @model_validator(mode="after")
    def require_canonical_source_axes(self) -> Self:
        if len(self.polytope.space.axes) != MAX_POLYTOPE_FACE_LATTICE_DIMENSION:
            raise _validation_error(
                "face_lattice_dimension",
                "polytope face lattices are defined here for dimension three",
            )
        if tuple(
            sorted(set(self.extreme_vertex_indices))
        ) != self.extreme_vertex_indices or any(
            index >= len(self.polytope.vertices)
            for index in self.extreme_vertex_indices
        ):
            raise _validation_error(
                "face_lattice_extreme_vertices",
                "extreme source vertex indices must be strictly increasing and in range",
            )
        face_keys = tuple(
            (face.dimension, face.source_vertex_indices) for face in self.faces
        )
        if tuple(sorted(set(face_keys))) != face_keys:
            raise _validation_error(
                "face_lattice_order",
                "faces must be unique and ordered by dimension and vertex indices",
            )
        if any(
            index >= len(self.polytope.vertices)
            for face in self.faces
            for index in face.source_vertex_indices
        ):
            raise _validation_error(
                "face_lattice_vertex_indices",
                "face vertex indices must refer to retained source vertices",
            )
        empty = PolytopeFace(dimension=-1, source_vertex_indices=())
        top = PolytopeFace(
            dimension=MAX_POLYTOPE_FACE_LATTICE_DIMENSION,
            source_vertex_indices=self.extreme_vertex_indices,
        )
        if (
            self.faces[0] != empty
            or self.faces[-1] != top
            or tuple(
                face.source_vertex_indices for face in self.faces if face.dimension == 0
            )
            != tuple((index,) for index in self.extreme_vertex_indices)
        ):
            raise _validation_error(
                "face_lattice_extremes",
                "the lattice must retain its unique bottom, top, and extreme vertices",
            )
        facet_labels = tuple(
            face.source_vertex_indices for face in self.faces if face.dimension == 2
        )
        if any(
            len(facet) < 3 or not set(facet).issubset(self.extreme_vertex_indices)
            for facet in facet_labels
        ):
            raise _validation_error(
                "face_lattice_facets",
                "facet labels must contain at least three extreme source vertices",
            )
        expected_face_keys, edges = _complete_face_lattice_labels(
            self.extreme_vertex_indices, facet_labels
        )
        if face_keys != expected_face_keys:
            raise _validation_error(
                "face_lattice_incomplete_faces",
                "the face labels must include every vertex, edge, facet, bottom, and top",
            )
        cover_pairs = tuple(
            (cover.lower_face_index, cover.upper_face_index) for cover in self.covers
        )
        if tuple(sorted(set(cover_pairs))) != cover_pairs or any(
            upper >= len(self.faces)
            or lower >= len(self.faces)
            or self.faces[upper].dimension != self.faces[lower].dimension + 1
            or not set(self.faces[lower].source_vertex_indices).issubset(
                self.faces[upper].source_vertex_indices
            )
            for lower, upper in cover_pairs
        ):
            raise _validation_error(
                "face_lattice_cover_relation",
                "cover relations must be unique, ordered, and respect face incidence",
            )
        expected_cover_pairs = _complete_face_lattice_covers(
            face_keys, self.extreme_vertex_indices, facet_labels, edges
        )
        if cover_pairs != expected_cover_pairs:
            raise _validation_error(
                "face_lattice_incomplete_covers",
                "covers must contain every dimension-adjacent face incidence",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polytope: RationalVPolytope,
        extreme_vertex_indices: tuple[int, ...],
        faces: tuple[PolytopeFace, ...],
        covers: tuple[PolytopeFaceCover, ...],
    ) -> Self:
        return cls.model_construct(
            polytope=polytope,
            extreme_vertex_indices=extreme_vertex_indices,
            faces=faces,
            covers=covers,
        )


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

    Admission uses output-sensitive upper bounds for the private exact
    double-description conversion. Complete axis-aligned boxes use their
    direct product structure. The field descriptions publish the ray and
    candidate-pair ceilings used by both representations.
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
            "After exact duplicate removal, admission applies McMullen's "
            "output bound to every incremental double-description prefix: "
            f"at most {MAX_DD_RAY_BOUND} rays, {MAX_DD_PAIR_BOUND} "
            "positive/negative candidate pairs, and a primitive-minor height "
            f"bound of {MAX_DD_COEFFICIENT_DIGITS} decimal digits, a pair-height "
            f"work ceiling of {MAX_DD_WEIGHTED_HEIGHT_WORK}, and at most "
            f"{MAX_PULLING_SIMPLEX_BOUND} incidence-driven pulling simplices. Complete two-level Cartesian "
            "boxes use a direct exact product conversion."
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
            "is rejected. Duplicate primitive rows are removed before the "
            "same output-sensitive double-description admission used by the "
            f"V form: at most {MAX_DD_RAY_BOUND} intermediate rays, "
            f"{MAX_DD_PAIR_BOUND} candidate pairs, and "
            f"{MAX_DD_COEFFICIENT_DIGITS} primitive-minor digits, "
            f"{MAX_DD_WEIGHTED_HEIGHT_WORK} pair-height work, and "
            f"{MAX_PULLING_SIMPLEX_BOUND} pulling simplices. Derived vertices reuse "
            "their exact incidence description for pulling triangulation."
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
    def accept_canonical_v_polytope_value(
        cls, data: object, info: ValidationInfo
    ) -> object:
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
            canonical = (
                RationalVPolytope.model_validate_json(json.dumps(value))
                if info.mode == "json"
                else RationalVPolytope.model_validate(value)
            )
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
            numerator_digits = max(
                numerator_digits, len(format_canonical_integer(abs(coord.num)))
            )
            denominator_digits = max(
                denominator_digits, len(format_canonical_integer(coord.den))
            )
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

    # Run the same retained conversion/triangulation admission used by the
    # native wrapper after exact deduplication.
    points, resolved_dim = _vertices_from_v_representation(vertices)
    prepared, triangulation = _prepare_volume_components(points, resolved_dim)
    return prepared, resolved_dim, triangulation


def _require_admissible_h_vertices(
    halfspaces: tuple[Halfspace, ...], dim: int
) -> tuple[list[list[Any]], list[tuple[int, ...]]]:
    """Admit the derived vertex set of an H-representation.

    One admitted homogeneous DD pass classifies emptiness, boundedness,
    lineality, affine dimension, vertices, and incidence. The retained
    incidences drive pulling triangulation without replaying conversion.
    """

    reduced = _deduplicate_halfspaces(halfspaces)
    rows = _halfspace_rows(reduced)
    conversion = halfspaces_to_generators(rows, dim)
    if conversion.empty:
        raise _validation_error(
            "h_representation", "the H-representation defines an empty polytope"
        )
    if not conversion.bounded:
        raise _validation_error(
            "halfspaces",
            "the H-representation is unbounded; polytope volume requires a bounded polytope",
        )
    verts = [
        [Rational(value.numerator, value.denominator) for value in point]
        for point in conversion.vertices
    ]
    if conversion.affine_dimension < dim:
        return verts, []
    facet_incidence = []
    for row_index in range(len(rows)):
        bits = 0
        for vertex_index, active in enumerate(conversion.vertex_incidence):
            if active & (1 << row_index):
                bits |= 1 << vertex_index
        if bits:
            facet_incidence.append(bits)
    facet_incidence = list(maximal_incidence_faces(facet_incidence))
    require_pulling_work_admissible(len(facet_incidence), dim)
    triangulation = list(pulling_triangulation(len(verts), dim, facet_incidence))
    if not triangulation:
        return verts, []
    _require_triangulated_volume_within_result_bound(
        [_point_digit_lengths(row) for row in verts], triangulation, dim
    )
    return verts, triangulation


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
                "halfspace_normal_zero", "half-space coefficients must not all be zero"
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


class PyramidBaseVertexMap(StrictModel):
    """One exact source-to-base vertex transport row of a pyramid construction."""

    source_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    pyramid_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)


class PyramidRequest(StrictModel):
    """Compute the exact pyramid over one bounded rational V-polytope.

    The pyramid embeds each base vertex ``p`` as ``(p, 0)`` on a fresh
    height axis and adds one apex ``(0, ..., 0, 1)``. The height axis must
    be a fresh label outside the source coordinate space, and the reserved
    apex vertex ID ``apex`` must not already occur among the source vertex
    IDs. The admitted envelope is one extra ambient dimension (at most
    ``MAX_RATIONAL_POLYTOPE_DIMENSION``) and one extra vertex row (at most
    ``MAX_VERTICES``); the construction itself adds no coordinate growth.
    """

    polytope: RationalVPolytope = Field(
        description=(
            "Nonempty bounded rational V-polytope serving as the pyramid base; "
            "base vertices keep their source IDs unchanged."
        )
    )
    height_axis: CoordinateAxis = Field(
        description=(
            "Fresh coordinate label carrying the pyramid height; it must not "
            "occur among the source space axes."
        )
    )


class PyramidResult(StrictModel):
    """Exact pyramid polytope with base/apex transport and dimension identity."""

    pyramid: RationalVPolytope = Field(
        description=(
            "Exact pyramid V-polytope on the source axes plus the height axis; "
            "base vertices carry last coordinate 0 and the apex carries 1."
        )
    )
    apex_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    base_vertex_map: tuple[PyramidBaseVertexMap, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "One transport row per source vertex, sorted by source vertex ID; "
            "base vertices retain their source IDs."
        ),
    )
    source_affine_dimension: int = Field(ge=0, le=MAX_FACET_DIMENSION)
    pyramid_affine_dimension: int = Field(ge=1, le=MAX_FACET_DIMENSION)

    @model_validator(mode="after")
    def require_pyramid_transport_shape(self) -> Self:
        if self.apex_vertex_id != "apex":
            raise _validation_error(
                "pyramid_apex_id",
                "the pyramid apex vertex ID is the reserved label 'apex'",
            )
        source_ids = tuple(row.source_vertex_id for row in self.base_vertex_map)
        if source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(
            source_ids
        ):
            raise _validation_error(
                "pyramid_transport_order",
                "base transport rows must be unique and sorted by source vertex ID",
            )
        pyramid_ids = tuple(vertex.vertex_id for vertex in self.pyramid.vertices)
        if self.apex_vertex_id not in pyramid_ids:
            raise _validation_error(
                "pyramid_apex_binding",
                "the apex vertex ID must occur among the pyramid vertices",
            )
        base_ids = tuple(row.pyramid_vertex_id for row in self.base_vertex_map)
        if tuple(sorted(base_ids)) != tuple(sorted(set(base_ids))) or len(
            base_ids
        ) != len(self.base_vertex_map):
            raise _validation_error(
                "pyramid_base_binding",
                "base transport targets must be distinct pyramid vertices",
            )
        if set(base_ids) | {self.apex_vertex_id} != set(pyramid_ids):
            raise _validation_error(
                "pyramid_vertex_cover",
                "base transport plus the apex must cover every pyramid vertex",
            )
        if self.pyramid_affine_dimension != self.source_affine_dimension + 1:
            raise _validation_error(
                "pyramid_dimension_identity",
                "pyramid affine dimension must be exactly source dimension plus one",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        pyramid: RationalVPolytope,
        base_vertex_map: tuple[PyramidBaseVertexMap, ...],
        source_affine_dimension: int,
        pyramid_affine_dimension: int,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its construction."""

        return cls.model_construct(
            pyramid=pyramid,
            apex_vertex_id="apex",
            base_vertex_map=base_vertex_map,
            source_affine_dimension=source_affine_dimension,
            pyramid_affine_dimension=pyramid_affine_dimension,
        )


class PrismVertexMap(StrictModel):
    """One exact source-to-prism vertex transport row."""

    source_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    prism_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    side: str = Field(description="Either 'bottom' (height 0) or 'top' (height 1).")

    @model_validator(mode="after")
    def require_known_side(self) -> Self:
        if self.side not in ("bottom", "top"):
            raise _validation_error(
                "prism_side", "prism transport side must be 'bottom' or 'top'"
            )
        return self


class PrismRequest(StrictModel):
    """Compute the exact prism ``P x [0, 1]`` over one rational V-polytope.

    Each source vertex ``p`` yields a bottom vertex ``(p, 0)`` and a top
    vertex ``(p, 1)`` on a fresh height axis. Transport IDs are
    ``f"{source_id}_bottom"`` and ``f"{source_id}_top"``; a source whose
    suffixed IDs collide or exceed the label bound is outside the admitted
    domain and the caller relabels first. The admitted envelope is one
    extra ambient dimension (at most ``MAX_RATIONAL_POLYTOPE_DIMENSION``)
    and twice the source vertex rows (at most ``MAX_VERTICES``); the
    construction itself adds no coordinate growth.
    """

    polytope: RationalVPolytope = Field(
        description=("Nonempty bounded rational V-polytope serving as the prism base.")
    )
    height_axis: CoordinateAxis = Field(
        description=(
            "Fresh coordinate label carrying the prism height; it must not "
            "occur among the source space axes."
        )
    )


class PrismResult(StrictModel):
    """Exact prism polytope with bottom/top transport and dimension identity."""

    prism: RationalVPolytope = Field(
        description=(
            "Exact prism V-polytope on the source axes plus the height axis; "
            "bottom vertices carry last coordinate 0 and top vertices carry 1."
        )
    )
    bottom_vertex_map: tuple[PrismVertexMap, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "One bottom transport row per source vertex, sorted by source vertex ID."
        ),
    )
    top_vertex_map: tuple[PrismVertexMap, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "One top transport row per source vertex, sorted by source vertex ID."
        ),
    )
    source_affine_dimension: int = Field(ge=0, le=MAX_FACET_DIMENSION)
    prism_affine_dimension: int = Field(ge=1, le=MAX_FACET_DIMENSION)

    @model_validator(mode="after")
    def require_prism_transport_shape(self) -> Self:
        for row in (*self.bottom_vertex_map, *self.top_vertex_map):
            if row.side not in ("bottom", "top"):
                raise _validation_error(
                    "prism_side", "prism transport side must be 'bottom' or 'top'"
                )
        if any(row.side != "bottom" for row in self.bottom_vertex_map):
            raise _validation_error(
                "prism_side", "bottom transport rows must all carry side 'bottom'"
            )
        if any(row.side != "top" for row in self.top_vertex_map):
            raise _validation_error(
                "prism_side", "top transport rows must all carry side 'top'"
            )
        bottom_sources = tuple(row.source_vertex_id for row in self.bottom_vertex_map)
        top_sources = tuple(row.source_vertex_id for row in self.top_vertex_map)
        if bottom_sources != tuple(sorted(bottom_sources)) or len(
            set(bottom_sources)
        ) != len(bottom_sources):
            raise _validation_error(
                "prism_transport_order",
                "bottom transport rows must be unique and sorted by source vertex ID",
            )
        if top_sources != tuple(sorted(top_sources)) or len(set(top_sources)) != len(
            top_sources
        ):
            raise _validation_error(
                "prism_transport_order",
                "top transport rows must be unique and sorted by source vertex ID",
            )
        if set(bottom_sources) != set(top_sources):
            raise _validation_error(
                "prism_transport_cover",
                "bottom and top transport rows must cover the same source vertices",
            )
        prism_ids = tuple(vertex.vertex_id for vertex in self.prism.vertices)
        bottom_ids = tuple(row.prism_vertex_id for row in self.bottom_vertex_map)
        top_ids = tuple(row.prism_vertex_id for row in self.top_vertex_map)
        if len(set(bottom_ids)) != len(bottom_ids) or len(set(top_ids)) != len(top_ids):
            raise _validation_error(
                "prism_binding",
                "prism transport targets must be distinct within each side",
            )
        if set(bottom_ids) & set(top_ids):
            raise _validation_error(
                "prism_binding",
                "bottom and top prism vertex IDs must be disjoint",
            )
        if set(bottom_ids) | set(top_ids) != set(prism_ids):
            raise _validation_error(
                "prism_vertex_cover",
                "bottom plus top transport must cover every prism vertex",
            )
        if self.prism_affine_dimension != self.source_affine_dimension + 1:
            raise _validation_error(
                "prism_dimension_identity",
                "prism affine dimension must be exactly source dimension plus one",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        prism: RationalVPolytope,
        bottom_vertex_map: tuple[PrismVertexMap, ...],
        top_vertex_map: tuple[PrismVertexMap, ...],
        source_affine_dimension: int,
        prism_affine_dimension: int,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its construction."""

        return cls.model_construct(
            prism=prism,
            bottom_vertex_map=bottom_vertex_map,
            top_vertex_map=top_vertex_map,
            source_affine_dimension=source_affine_dimension,
            prism_affine_dimension=prism_affine_dimension,
        )


class JoinVertexMap(StrictModel):
    """One exact source-to-join vertex transport row."""

    source_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    join_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    side: str = Field(description="Either 'left' or 'right'.")

    @model_validator(mode="after")
    def require_known_side(self) -> Self:
        if self.side not in ("left", "right"):
            raise _validation_error(
                "join_side", "join transport side must be 'left' or 'right'"
            )
        return self


class JoinRequest(StrictModel):
    """Compute the exact join ``P * Q`` of two rational V-polytopes.

    The factors must live on disjoint axis labels and carry disjoint
    vertex IDs; the height axis must be fresh outside both spaces. The
    left factor embeds as ``(p, 0, 0)`` and the right factor as
    ``(0, q, 1)`` on the output axes
    ``(*left.axes, *right.axes, height_axis)``, keeping source vertex IDs
    unchanged. The admitted envelope is the combined ambient dimension
    (at most ``MAX_RATIONAL_POLYTOPE_DIMENSION``) and the combined vertex
    rows (at most ``MAX_VERTICES``); the construction itself adds no
    coordinate growth.
    """

    left: RationalVPolytope = Field(
        description="Left join factor; axes and vertex IDs must be disjoint from right."
    )
    right: RationalVPolytope = Field(
        description="Right join factor; axes and vertex IDs must be disjoint from left."
    )
    height_axis: CoordinateAxis = Field(
        description=(
            "Fresh coordinate label carrying the join height; it must not "
            "occur among either factor's space axes."
        )
    )


class JoinResult(StrictModel):
    """Exact join polytope with left/right transport and dimension identity."""

    join: RationalVPolytope = Field(
        description=(
            "Exact join V-polytope on (*left.axes, *right.axes, height_axis); "
            "left vertices carry right-block 0 and height 0, right vertices "
            "carry left-block 0 and height 1."
        )
    )
    left_vertex_map: tuple[JoinVertexMap, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description="One transport row per left vertex, sorted by source vertex ID.",
    )
    right_vertex_map: tuple[JoinVertexMap, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description="One transport row per right vertex, sorted by source vertex ID.",
    )
    left_affine_dimension: int = Field(ge=0, le=MAX_FACET_DIMENSION)
    right_affine_dimension: int = Field(ge=0, le=MAX_FACET_DIMENSION)
    join_affine_dimension: int = Field(ge=1, le=MAX_FACET_DIMENSION)

    @model_validator(mode="after")
    def require_join_transport_shape(self) -> Self:
        if any(row.side != "left" for row in self.left_vertex_map):
            raise _validation_error(
                "join_side", "left transport rows must all carry side 'left'"
            )
        if any(row.side != "right" for row in self.right_vertex_map):
            raise _validation_error(
                "join_side", "right transport rows must all carry side 'right'"
            )
        left_sources = tuple(row.source_vertex_id for row in self.left_vertex_map)
        right_sources = tuple(row.source_vertex_id for row in self.right_vertex_map)
        if left_sources != tuple(sorted(left_sources)) or len(set(left_sources)) != len(
            left_sources
        ):
            raise _validation_error(
                "join_transport_order",
                "left transport rows must be unique and sorted by source vertex ID",
            )
        if right_sources != tuple(sorted(right_sources)) or len(
            set(right_sources)
        ) != len(right_sources):
            raise _validation_error(
                "join_transport_order",
                "right transport rows must be unique and sorted by source vertex ID",
            )
        join_ids = tuple(vertex.vertex_id for vertex in self.join.vertices)
        left_ids = tuple(row.join_vertex_id for row in self.left_vertex_map)
        right_ids = tuple(row.join_vertex_id for row in self.right_vertex_map)
        if set(left_ids) & set(right_ids):
            raise _validation_error(
                "join_binding",
                "left and right join vertex IDs must be disjoint",
            )
        if set(left_ids) | set(right_ids) != set(join_ids):
            raise _validation_error(
                "join_vertex_cover",
                "left plus right transport must cover every join vertex",
            )
        if self.join_affine_dimension != (
            self.left_affine_dimension + self.right_affine_dimension + 1
        ):
            raise _validation_error(
                "join_dimension_identity",
                "join affine dimension must be left plus right plus one",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        join: RationalVPolytope,
        left_vertex_map: tuple[JoinVertexMap, ...],
        right_vertex_map: tuple[JoinVertexMap, ...],
        left_affine_dimension: int,
        right_affine_dimension: int,
        join_affine_dimension: int,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its construction."""

        return cls.model_construct(
            join=join,
            left_vertex_map=left_vertex_map,
            right_vertex_map=right_vertex_map,
            left_affine_dimension=left_affine_dimension,
            right_affine_dimension=right_affine_dimension,
            join_affine_dimension=join_affine_dimension,
        )


class PolytopeEdge(StrictModel):
    """One undirected edge of a labelled rational polytope edge profile.

    Endpoints are source vertex IDs with ``endpoint_a < endpoint_b`` so every
    edge has one canonical serialization; loops are never edges.
    """

    endpoint_a: str = Field(
        min_length=1,
        max_length=MAX_COORDINATE_LABEL_LENGTH,
    )
    endpoint_b: str = Field(
        min_length=1,
        max_length=MAX_COORDINATE_LABEL_LENGTH,
    )

    @model_validator(mode="after")
    def require_ordered_endpoints(self) -> Self:
        for endpoint in (self.endpoint_a, self.endpoint_b):
            _require_unicode_scalar_label(endpoint)
        if not self.endpoint_a < self.endpoint_b:
            raise _validation_error(
                "edge_endpoints",
                "edge endpoints must satisfy endpoint_a < endpoint_b",
            )
        return self


MAX_EDGE_PROFILE_EDGES = 2016
"""Maximum number of edges materialized by one edge-profile result.

The complete graph on ``MAX_VERTICES = 64`` rows has ``64 * 63 / 2``
pairs; no profile reports more.
"""


class EdgeProfileRequest(StrictModel):
    """Compute the exact vertex-adjacency (edge) graph of one V-polytope.

    The source must be full-dimensional: the points must affinely span
    their ambient dimension, exactly as the facet operation requires,
    because the kernel reuses its output-sensitive DD conversion. Repeated
    source rows are retained for incidence binding but collapse before
    conversion; exact facet incidences then drive the bounded edge pass.
    """

    polytope: RationalVPolytope = Field(
        description=(
            "Full-dimensional labelled rational V-polytope whose edge graph "
            "is computed; redundant boundary rows carry no edges."
        )
    )
    dimension_bound: int = Field(
        default=MAX_FACET_DIMENSION,
        ge=1,
        le=MAX_FACET_DIMENSION,
        description="Maximum admitted ambient dimension for this edge profile.",
    )


class EdgeProfileResult(StrictModel):
    """Exact edge graph of a labelled rational polytope with dimension replay."""

    polytope: RationalVPolytope = Field(
        description="The exact retained source V-representation.",
    )
    edges: tuple[PolytopeEdge, ...] = Field(
        min_length=0,
        max_length=MAX_EDGE_PROFILE_EDGES,
        description=(
            "All undirected vertex-adjacency edges over the exact extreme "
            "vertices, sorted lexicographically by endpoint pair."
        ),
    )
    edge_count: int = Field(
        ge=0,
        le=MAX_EDGE_PROFILE_EDGES,
        description="Number of reported edges; always equals ``len(edges)``.",
    )
    affine_dimension: int = Field(
        ge=0,
        le=MAX_FACET_DIMENSION,
        description="Replayed exact affine dimension of the source polytope.",
    )

    @model_validator(mode="after")
    def require_edge_profile_shape(self) -> Self:
        if self.edge_count != len(self.edges):
            raise _validation_error(
                "edge_count",
                "edge_count must equal the number of reported edges",
            )
        pairs = tuple((edge.endpoint_a, edge.endpoint_b) for edge in self.edges)
        if pairs != tuple(sorted(pairs)) or len(set(pairs)) != len(pairs):
            raise _validation_error(
                "edge_order",
                "edges must be unique and sorted lexicographically",
            )
        source_ids = {vertex.vertex_id for vertex in self.polytope.vertices}
        if any(endpoint not in source_ids for pair in pairs for endpoint in pair):
            raise _validation_error(
                "edge_binding",
                "every edge endpoint must occur among the retained polytope vertices",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        polytope: RationalVPolytope,
        edges: tuple[PolytopeEdge, ...],
        affine_dimension: int,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its enumeration."""

        return cls.model_construct(
            polytope=polytope,
            edges=edges,
            edge_count=len(edges),
            affine_dimension=affine_dimension,
        )


class VertexFigureVertexMap(StrictModel):
    """One exact neighbor-to-figure vertex transport row of a vertex figure."""

    source_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    figure_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)


class VertexFigureRequest(StrictModel):
    """Compute the exact vertex figure of one polytope vertex.

    The vertex figure at ``vertex_id`` is the convex hull of the
    edge-midpoints ``(v + u) / 2`` over the edge neighbors ``u`` of ``v``.
    The center must be an exact extreme vertex of a polytope of affine
    dimension at least one; redundant source rows and dimension-zero
    sources are outside the admitted domain. Figure vertex IDs are
    derived deterministically as ``sec_<neighbor_id>`` and must respect
    the label bound, so an over-long neighbor ID is rejected and the
    caller relabels first. The admitted envelope mirrors the edge
    profile's output-sensitive DD and result bounds.
    """

    polytope: RationalVPolytope = Field(
        description=(
            "Full-dimensional labelled rational V-polytope carrying the center vertex."
        )
    )
    vertex_id: CoordinateAxis = Field(
        description="Vertex ID of the center whose figure is computed.",
    )


class VertexFigurePolytope(StrictModel):
    """A labelled exact vertex-figure V-representation on its source axes.

    Unlike the canonical full-dimensional ``RationalVPolytope`` value, a
    vertex figure of a ``d``-polytope is ``(d - 1)``-dimensional in the
    same ambient space, so it cannot satisfy the full-dimensional count
    invariant. Its IDs are still strictly ordered with distinct
    coordinates of the declared axis length; extremality and the
    dimension identity are established by the owning vertex-figure
    operation rather than when this value is parsed.
    """

    space: RationalCoordinateSpace
    vertices: tuple[RationalPolytopeVertex, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
    )

    @model_validator(mode="after")
    def require_canonical_figure_vertices(self) -> Self:
        dimension = len(self.space.axes)
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


class VertexFigureResult(StrictModel):
    """Exact vertex figure with neighbor transport and dimension identity."""

    figure: VertexFigurePolytope = Field(
        description=(
            "Exact vertex-figure V-polytope on the source axes; every vertex "
            "is the midpoint (v + u) / 2 of the center v with one edge "
            "neighbor u."
        )
    )
    center_vertex_id: str = Field(min_length=1, max_length=MAX_COORDINATE_LABEL_LENGTH)
    vertex_map: tuple[VertexFigureVertexMap, ...] = Field(
        min_length=1,
        max_length=MAX_VERTICES,
        description=(
            "One transport row per edge neighbor of the center, sorted by "
            "source vertex ID; figure IDs are `sec_<neighbor_id>`."
        ),
    )
    source_affine_dimension: int = Field(ge=1, le=MAX_FACET_DIMENSION)
    figure_affine_dimension: int = Field(ge=0, le=MAX_FACET_DIMENSION)

    @model_validator(mode="after")
    def require_vertex_figure_transport_shape(self) -> Self:
        source_ids = tuple(row.source_vertex_id for row in self.vertex_map)
        if source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(
            source_ids
        ):
            raise _validation_error(
                "vertex_figure_transport_order",
                "vertex-figure transport rows must be unique and sorted "
                "by source vertex ID",
            )
        for row in self.vertex_map:
            _require_unicode_scalar_label(row.source_vertex_id)
            _require_unicode_scalar_label(row.figure_vertex_id)
            if row.figure_vertex_id != f"sec_{row.source_vertex_id}":
                raise _validation_error(
                    "vertex_figure_id_derivation",
                    "figure vertex IDs must be exactly `sec_<neighbor_id>`",
                )
        figure_ids = tuple(vertex.vertex_id for vertex in self.figure.vertices)
        mapped_ids = tuple(row.figure_vertex_id for row in self.vertex_map)
        if set(mapped_ids) != set(figure_ids) or len(mapped_ids) != len(figure_ids):
            raise _validation_error(
                "vertex_figure_vertex_cover",
                "neighbor transport must cover every figure vertex exactly once",
            )
        if self.center_vertex_id in set(figure_ids) | set(source_ids):
            raise _validation_error(
                "vertex_figure_center_binding",
                "the center vertex ID must not occur among the figure "
                "vertices or neighbor IDs",
            )
        if self.figure_affine_dimension != self.source_affine_dimension - 1:
            raise _validation_error(
                "vertex_figure_dimension_identity",
                "figure affine dimension must be exactly source dimension minus one",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        figure: VertexFigurePolytope,
        center_vertex_id: str,
        vertex_map: tuple[VertexFigureVertexMap, ...],
        source_affine_dimension: int,
        figure_affine_dimension: int,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its construction."""

        return cls.model_construct(
            figure=figure,
            center_vertex_id=center_vertex_id,
            vertex_map=vertex_map,
            source_affine_dimension=source_affine_dimension,
            figure_affine_dimension=figure_affine_dimension,
        )


__all__ = [
    "MAX_COMPUTED_FACETS",
    "MAX_DIMENSION",
    "MAX_EDGE_PROFILE_EDGES",
    "MAX_EXTREMALITY_HEIGHT_WORK",
    "MAX_FACETS",
    "MAX_FACET_COORDINATE_DIGITS",
    "MAX_FACET_DIMENSION",
    "MAX_FACET_INCIDENCES",
    "MAX_POLYTOPE_FACE_LATTICE_COVERS",
    "MAX_POLYTOPE_FACE_LATTICE_DIMENSION",
    "MAX_POLYTOPE_FACE_LATTICE_EDGES",
    "MAX_POLYTOPE_FACE_LATTICE_FACES",
    "MAX_POLYTOPE_FACE_LATTICE_FACETS",
    "MAX_POLYTOPE_FACE_LATTICE_RESULT_CHARS",
    "MAX_POLYTOPE_FACE_LATTICE_WORK",
    "MAX_SUPPORT_COMPONENT_DIGITS",
    "MAX_VERTICES",
    "EdgeProfileRequest",
    "EdgeProfileResult",
    "FacetIncidenceRequest",
    "FacetIncidenceResult",
    "Halfspace",
    "JoinRequest",
    "JoinResult",
    "JoinVertexMap",
    "PolytopeEdge",
    "PolytopeFace",
    "PolytopeFaceCover",
    "PolytopeFaceLatticeRequest",
    "PolytopeFaceLatticeResult",
    "PolytopeSupportRequest",
    "PolytopeSupportResult",
    "PolytopeVolumeRequest",
    "PolytopeVolumeResult",
    "PrimitiveFacet",
    "PrismRequest",
    "PrismResult",
    "PrismVertexMap",
    "PyramidBaseVertexMap",
    "PyramidRequest",
    "PyramidResult",
    "RationalCoordinateSpace",
    "RationalCovector",
    "RationalExposedFace",
    "RationalPolytopeVertex",
    "RationalVPolytope",
    "Vertex",
    "VertexFigurePolytope",
    "VertexFigureRequest",
    "VertexFigureResult",
    "VertexFigureVertexMap",
]
