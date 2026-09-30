"""Typed contracts for exact permutation-valued lattice-gauge holonomy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self, TypeGuard

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.groups._table_models import (
    MAX_FINITE_TABLE_GROUP_ORDER,
    FiniteGroupTable,
    FiniteGroupTableElement,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"lattice_gauge.{reason}", message)


def _has_canonical_steps(steps: object) -> TypeGuard[tuple[GaugePathStep, ...]]:
    return (
        isinstance(steps, tuple)
        and type(steps) is tuple
        and len(steps) <= MAX_GAUGE_PATH_LENGTH
        and all(
            type(step) is GaugePathStep
            and _is_raw_label(getattr(step, "edge_id", None))
            and type(getattr(step, "forward", None)) is bool
            for step in steps
        )
    )


def _has_canonical_path_steps(path: object) -> bool:
    return _has_canonical_steps(getattr(path, "steps", None))


def _raw_fields(
    value: object,
    models: tuple[type[StrictModel], ...],
    allowed: set[str],
    code: str,
) -> dict[str, object]:
    """Inspect a closed raw object or canonical native carrier without copying."""
    if isinstance(value, dict) and type(value) is dict:
        fields = value
    elif isinstance(value, StrictModel) and type(value) in models:
        fields = value.__dict__
    else:
        raise _validation_error(code, "expected a canonical object with named fields")
    if len(fields) > len(allowed) or any(
        type(key) is not str or key not in allowed for key in fields
    ):
        raise _validation_error(code, "object has unknown fields")
    return fields


def _raw_sequence(
    value: object, limit: int, code: str
) -> tuple[object, ...] | list[object]:
    if (
        not isinstance(value, (tuple, list))
        or type(value) not in (tuple, list)
        or len(value) > limit
    ):
        raise _validation_error(
            code, f"expected a sequence with at most {limit} entries"
        )
    return value


def _check_raw_complex_shape(value: object) -> None:
    """Bound every nested container before canonical JSON copying begins."""
    fields = _raw_fields(
        value,
        (FiniteGroupGaugeComplex, FiniteGroupGaugeComplexRequest),
        {"lattice", "group", "faces"},
        "complex_request_shape",
    )
    _check_raw_lattice_shape(fields.get("lattice"))
    _check_raw_group_shape(fields.get("group"))
    _check_raw_faces_shape(fields.get("faces"))


def _check_raw_lattice_shape(lattice: object) -> None:
    fields = _raw_fields(
        lattice, (GaugeLattice,), {"vertices", "edges"}, "complex_lattice_shape"
    )
    vertices = _raw_sequence(
        fields.get("vertices"), MAX_GAUGE_VERTICES, "complex_vertex_bound"
    )
    edges = _raw_sequence(fields.get("edges"), MAX_GAUGE_EDGES, "complex_edge_bound")
    if any(not _is_raw_label(vertex) for vertex in vertices):
        raise _validation_error(
            "complex_vertex_shape", "each vertex must be a gauge label"
        )
    for edge in edges:
        edge_fields = _raw_fields(
            edge, (GaugeEdge,), {"edge_id", "tail", "head"}, "complex_edge_shape"
        )
        if any(
            not _is_raw_label(edge_fields.get(key))
            for key in ("edge_id", "tail", "head")
        ):
            raise _validation_error(
                "complex_edge_shape", "each edge must carry gauge labels"
            )


def _check_raw_group_shape(group: object) -> None:
    fields = _raw_fields(
        group,
        (FiniteGroupTable,),
        {"multiplication", "identity", "inverse"},
        "complex_group_shape",
    )
    rows = _raw_sequence(
        fields.get("multiplication"),
        MAX_FINITE_TABLE_GROUP_ORDER,
        "complex_group_bound",
    )
    inverse = _raw_sequence(
        fields.get("inverse"), MAX_FINITE_TABLE_GROUP_ORDER, "complex_group_bound"
    )
    for row in rows:
        for entry in _raw_sequence(
            row, MAX_FINITE_TABLE_GROUP_ORDER, "complex_group_bound"
        ):
            _check_raw_group_index(entry)
    _check_raw_group_index(fields.get("identity"))
    for entry in inverse:
        _check_raw_group_index(entry)


def _check_raw_group_index(value: object) -> None:
    if type(value) is not int or not 0 <= value < MAX_FINITE_TABLE_GROUP_ORDER:
        raise _validation_error(
            "complex_group_shape", "group entries must be bounded integer indices"
        )


def _check_raw_faces_shape(faces: object) -> None:
    faces = _raw_sequence(faces, MAX_GAUGE_FACES, "complex_face_bound")
    total_steps = 0
    for face in faces:
        total_steps += _raw_face_step_count(face)
        if total_steps > MAX_GAUGE_TOTAL_FACE_STEPS:
            raise _validation_error(
                "complex_boundary_bound",
                "aggregate face boundary exceeds the finite complex envelope",
            )


def _raw_face_step_count(face: object) -> int:
    fields = _raw_fields(
        face, (FiniteGroupGaugeFace,), {"face_id", "boundary"}, "complex_face_shape"
    )
    if not _is_raw_label(fields.get("face_id")):
        raise _validation_error(
            "complex_face_shape", "each face must be a labelled walk"
        )
    return _raw_path_step_count(fields.get("boundary"))


def _raw_path_step_count(value: object) -> int:
    boundary = _raw_fields(
        value,
        (OrientedGaugePath,),
        {"steps", "basepoint"},
        "complex_boundary_shape",
    )
    steps = _raw_sequence(
        boundary.get("steps"), MAX_GAUGE_PATH_LENGTH, "complex_face_length"
    )
    basepoint = boundary.get("basepoint")
    if basepoint is not None and not _is_raw_label(basepoint):
        raise _validation_error(
            "complex_boundary_shape", "a boundary basepoint must be a gauge label"
        )
    for step in steps:
        step_fields = _raw_fields(
            step, (GaugePathStep,), {"edge_id", "forward"}, "complex_step_shape"
        )
        if (
            not _is_raw_label(step_fields.get("edge_id"))
            or type(step_fields.get("forward")) is not bool
        ):
            raise _validation_error(
                "complex_step_shape", "face steps must be labelled boolean traversals"
            )
    return len(steps)


def _lattice_walk_endpoints(
    by_id: Mapping[str, GaugeEdge],
    steps: object,
) -> tuple[str | None, str | None]:
    """Return the first and last vertex of a walk over the retained lattice.

    Every step must name a lattice edge, traverse it in one of its two
    directions, and continue from the previous head. A consumer reads these
    walks directly, so a decoded path must be one the lattice admits.
    """
    if not _has_canonical_steps(steps):
        raise _validation_error(
            "lattice_walk", "walk steps must be bounded canonical lattice traversals"
        )
    cursor: str | None = None
    first: str | None = None
    for step in steps:
        edge = by_id.get(step.edge_id)
        if edge is None or type(step.forward) is not bool:
            raise _validation_error(
                "lattice_walk", "walk steps must be typed lattice traversals"
            )
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and tail != cursor:
            raise _validation_error(
                "lattice_walk", "walk steps must chain head-to-tail"
            )
        if first is None:
            first = tail
        cursor = head
    return first, cursor


def _is_raw_label(value: object) -> bool:
    """One retained label is a bounded string, not a container of strings."""
    return type(value) is str and 1 <= len(value) <= MAX_GAUGE_LABEL_LENGTH


MAX_GAUGE_VERTICES = 64
"""Maximum vertices in one admitted gauge lattice."""

MAX_GAUGE_EDGES = 128
"""Maximum canonically oriented edges in one admitted gauge lattice."""

MAX_GAUGE_DEGREE = 8
"""Maximum permutation degree of the structure group S_d."""

MIN_GAUGE_DEGREE = 1
"""Minimum permutation degree; degree one is the trivial structure group."""

MAX_GAUGE_PATH_LENGTH = 256
"""Maximum oriented steps in one admitted lattice path."""

MAX_GAUGE_FACES = 128
"""Maximum oriented 2-cells in one admitted finite gauge complex."""

MAX_GAUGE_TOTAL_FACE_STEPS = 4096
"""Maximum aggregate attaching-walk steps in one gauge complex."""


MAX_GAUGE_LOOP_FAMILY_SIZE = 128
"""Maximum number of explicitly supplied permutation-valued loops."""

MAX_GAUGE_LOOP_FAMILY_STEPS = 4096
"""Maximum aggregate path steps evaluated in one loop family."""

MAX_GAUGE_LOOP_FAMILY_WORK = 750_000
"""Maximum admitted field, path, and permutation work for one loop family."""

MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS = 350_000
"""Maximum value cells and scalar text units for one loop family result."""
MAX_FINITE_GROUP_GAUGE_COMPLEX_OUTPUT_UNITS = 5_592
"""Maximum materialized value cells retained by one finite gauge complex.

This is an operation-owned structural bound, not an encoded-size estimate.
Every label is separately bounded by ``MAX_GAUGE_LABEL_LENGTH``, so charging
characters here would make a delivery-oriented width part of the native
mathematical domain; encoded-byte limits belong to the delivery boundary.

The value is the structural worst case of the per-component bounds
(64 base + 64 vertices + 3*128 edges + 2*128 faces + 4096 face steps +
128 explicit face basepoints + 24**2 table cells + 24 elements). Both
nonconstant and constant faces may retain a basepoint, so all 128 can coexist
with the maximum step count. This aggregate check is a defence-in-depth guard
rather than the primary limit.
"""

# Materialized-cell weights for one finite gauge complex. Each retained value
# is one cell: a vertex, one of an edge's three label references, a face, or
# one oriented face step.
_GAUGE_COMPLEX_BASE_CELLS = 64
_GAUGE_CELLS_PER_VERTEX = 1
_GAUGE_CELLS_PER_EDGE = 3
_GAUGE_CELLS_PER_FACE = 2
_GAUGE_CELLS_PER_GROUP_ELEMENT = 1

MAX_GAUGE_LABEL_LENGTH = 64
"""Maximum length of a vertex or edge identifier."""


def _require_scalar_label(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _validation_error(
            "unicode_scalar_label", "labels must contain only Unicode scalar values"
        )
    return value


GaugeLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_GAUGE_LABEL_LENGTH, strict=True),
    AfterValidator(_require_scalar_label),
]
"""One vertex or edge identifier in a finite gauge lattice."""


class GaugeEdge(StrictModel):
    """One canonically oriented lattice edge with stable identity."""

    edge_id: GaugeLabel
    tail: GaugeLabel
    head: GaugeLabel


class GaugeLattice(StrictModel):
    """A finite vertex domain with canonically oriented edge IDs."""

    vertices: tuple[GaugeLabel, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )
    edges: tuple[GaugeEdge, ...] = Field(min_length=1, max_length=MAX_GAUGE_EDGES)

    @model_validator(mode="after")
    def require_canonical_lattice(self) -> Self:
        if len(set(self.vertices)) != len(self.vertices):
            raise _validation_error(
                "lattice_vertices_unique", "lattice vertices must be unique"
            )
        edge_ids = tuple(edge.edge_id for edge in self.edges)
        if tuple(sorted(edge_ids)) != edge_ids or len(set(edge_ids)) != len(edge_ids):
            raise _validation_error(
                "lattice_edge_ids",
                "lattice edge IDs must be unique and strictly ordered",
            )
        known = set(self.vertices)
        if any(edge.tail not in known or edge.head not in known for edge in self.edges):
            raise _validation_error(
                "lattice_endpoints",
                "every edge endpoint must be a lattice vertex",
            )
        return self


class PermutationLabel(StrictModel):
    """One exact permutation of ``0..degree-1`` acting on the right.

    The image tuple sends each point to its image; composition applies
    left-to-right along traversal order. Degree one represents the trivial
    group and has the unique identity permutation ``(0,)``.
    """

    degree: StrictInt = Field(ge=MIN_GAUGE_DEGREE, le=MAX_GAUGE_DEGREE)
    image: tuple[StrictInt, ...] = Field(
        min_length=MIN_GAUGE_DEGREE, max_length=MAX_GAUGE_DEGREE
    )

    @model_validator(mode="after")
    def require_permutation_shape(self) -> Self:
        if len(self.image) != self.degree:
            raise _validation_error(
                "permutation_shape",
                "permutation image must carry exactly degree entries",
            )
        if sorted(self.image) != list(range(self.degree)):
            raise _validation_error(
                "permutation_bijective",
                "permutation image must be a bijection of 0..degree-1",
            )
        return self


class GaugeFieldEdgeLabel(StrictModel):
    """The exact group element on one canonically oriented edge."""

    edge_id: GaugeLabel
    label: PermutationLabel


class GaugeField(StrictModel):
    """One exact permutation-valued edge field over a lattice.

    Every canonically oriented edge carries exactly one label; reverse
    traversal uses the exact inverse and is never submitted independently.
    """

    lattice: GaugeLattice
    degree: StrictInt = Field(ge=MIN_GAUGE_DEGREE, le=MAX_GAUGE_DEGREE)
    edge_labels: tuple[GaugeFieldEdgeLabel, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_EDGES
    )

    @model_validator(mode="after")
    def require_complete_edge_field(self) -> Self:
        if any(label.label.degree != self.degree for label in self.edge_labels):
            raise _validation_error(
                "field_degree",
                "every edge label must use the field structure degree",
            )
        have = tuple(label.edge_id for label in self.edge_labels)
        want = tuple(edge.edge_id for edge in self.lattice.edges)
        if tuple(sorted(have)) != tuple(sorted(set(have))) or set(have) != set(want):
            raise _validation_error(
                "field_coverage",
                "edge labels must cover every lattice edge exactly once",
            )
        return self


class GaugePathStep(StrictModel):
    """One oriented traversal of a lattice edge."""

    edge_id: GaugeLabel
    forward: StrictBool


class OrientedGaugePath(StrictModel):
    """An ordered edge walk, or a based zero-length identity path."""

    steps: tuple[GaugePathStep, ...] = Field(max_length=MAX_GAUGE_PATH_LENGTH)
    basepoint: GaugeLabel | None = None

    @model_validator(mode="after")
    def require_basepoint_for_empty_path(self) -> Self:
        basepoint = getattr(self, "basepoint", None)
        if not _has_canonical_path_steps(self) or (
            basepoint is not None and not _is_raw_label(basepoint)
        ):
            raise _validation_error(
                "path_shape", "paths must retain bounded canonical steps and basepoints"
            )
        if not self.steps and basepoint is None:
            raise _validation_error(
                "empty_path_basepoint",
                "a zero-length path must name its identity-path basepoint",
            )
        return self


class FiniteGroupGaugeEdgeLabel(StrictModel):
    """One edge value as an element of a specific finite multiplication table."""

    edge_id: GaugeLabel
    value: FiniteGroupTableElement


class FiniteGroupGaugeField(StrictModel):
    """A finite-table-valued field; its table is the coefficient group parent."""

    lattice: GaugeLattice
    group: FiniteGroupTable
    edge_values: tuple[FiniteGroupGaugeEdgeLabel, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_EDGES
    )

    @model_validator(mode="after")
    def require_edge_coverage(self) -> Self:
        group = getattr(self, "group", None)
        lattice = getattr(self, "lattice", None)
        edge_values = getattr(self, "edge_values", None)
        if (
            not isinstance(group, FiniteGroupTable)
            or type(getattr(group, "multiplication", None)) is not tuple
            or not isinstance(lattice, GaugeLattice)
            or type(getattr(lattice, "edges", None)) is not tuple
            or type(edge_values) is not tuple
        ):
            raise _validation_error(
                "finite_group_field_coverage",
                "finite-group field must retain its table, lattice, and edge values",
            )
        order = len(group.multiplication)
        if any(
            not isinstance(getattr(value, "value", None), FiniteGroupTableElement)
            or getattr(value.value, "group", None) != group
            or type(getattr(value.value, "index", None)) is not int
            or not 0 <= getattr(value.value, "index", -1) < order
            for value in edge_values
        ):
            raise _validation_error(
                "finite_group_value_parent",
                "every edge element must use the field's exact table parent",
            )
        have: tuple[str, ...] = tuple(
            str(getattr(value, "edge_id", "")) for value in edge_values
        )
        want: tuple[str, ...] = tuple(
            str(getattr(edge, "edge_id", "")) for edge in lattice.edges
        )
        if (
            "" in have
            or "" in want
            or tuple(sorted(have)) != tuple(sorted(set(have)))
            or set(have) != set(want)
        ):
            raise _validation_error(
                "finite_group_field_coverage",
                "edge values must cover every lattice edge once",
            )
        return self


class FiniteGroupGaugeFace(StrictModel):
    """One oriented 2-cell attached by a closed edge word.

    The ordered walk is the attaching map of the oriented face. Reversing the
    face reverses the walk and flips every step orientation. A constant
    attaching map is represented by an empty walk at its explicit basepoint.
    """

    face_id: GaugeLabel
    boundary: OrientedGaugePath


class FiniteGroupGaugeComplex(StrictModel):
    """A finite oriented 2-complex bound to one lattice and finite group."""

    lattice: GaugeLattice
    group: FiniteGroupTable
    faces: tuple[FiniteGroupGaugeFace, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_FACES
    )

    @model_validator(mode="before")
    @classmethod
    def canonicalize_json_arrays(cls, value: object) -> object:
        _check_raw_complex_shape(value)
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_closed_source_bound_faces(self) -> Self:
        edges = {edge.edge_id: edge for edge in self.lattice.edges}
        face_ids = tuple(face.face_id for face in self.faces)
        if tuple(sorted(face_ids)) != face_ids or len(set(face_ids)) != len(face_ids):
            raise _validation_error(
                "complex_face_ids", "face IDs must be unique and strictly ordered"
            )
        total_steps = 0
        for face in self.faces:
            path = face.boundary
            steps = path.steps
            if len(steps) > MAX_GAUGE_PATH_LENGTH:
                raise _validation_error(
                    "complex_face_length", "face boundary exceeds 256 oriented steps"
                )
            total_steps += len(steps)
            if total_steps > MAX_GAUGE_TOTAL_FACE_STEPS:
                raise _validation_error(
                    "complex_boundary_bound",
                    "aggregate face boundary exceeds the finite complex envelope",
                )
            if not steps:
                if path.basepoint not in self.lattice.vertices:
                    raise _validation_error(
                        "complex_empty_face_basepoint",
                        "a constant face attachment must name a source lattice vertex",
                    )
                continue
            first: str | None = None
            cursor: str | None = None
            for step in steps:
                edge = edges.get(step.edge_id)
                if edge is None:
                    raise _validation_error(
                        "complex_face_edge",
                        "face boundary must use source lattice edges",
                    )
                tail, head = (
                    (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
                )
                if cursor is not None and cursor != tail:
                    raise _validation_error(
                        "complex_face_chain",
                        "face boundary steps must chain head-to-tail",
                    )
                if first is None:
                    first = tail
                cursor = head
            if first != cursor or (
                path.basepoint is not None and path.basepoint != first
            ):
                raise _validation_error(
                    "complex_face_closed", "each oriented face boundary must be closed"
                )
        # Materialized cells, not encoded characters. Label widths are already
        # bounded by MAX_GAUGE_LABEL_LENGTH, so charging characters here would
        # make a delivery-oriented width part of the mathematical contract.
        order = len(self.group.multiplication)
        output_units = (
            _GAUGE_COMPLEX_BASE_CELLS
            + order * order
            + order * _GAUGE_CELLS_PER_GROUP_ELEMENT
            + len(self.lattice.vertices) * _GAUGE_CELLS_PER_VERTEX
            + len(self.lattice.edges) * _GAUGE_CELLS_PER_EDGE
        )
        for face in self.faces:
            output_units += _GAUGE_CELLS_PER_FACE
            output_units += len(face.boundary.steps)
            output_units += int(face.boundary.basepoint is not None)
        if output_units > MAX_FINITE_GROUP_GAUGE_COMPLEX_OUTPUT_UNITS:
            raise _validation_error(
                "complex_output_bound",
                "source-bound complex exceeds the conservative output-unit envelope",
            )
        return self


class FiniteGroupGaugeComplexRequest(StrictModel):
    """Construct source-bound oriented face boundaries over one gauge lattice."""

    lattice: GaugeLattice
    group: FiniteGroupTable
    faces: tuple[FiniteGroupGaugeFace, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_FACES
    )

    @model_validator(mode="before")
    @classmethod
    def admit_raw_face_growth(cls, value: object) -> object:
        _check_raw_complex_shape(value)
        return canonicalize_json_containers(value)


class FiniteGroupGaugeCurvatureRequest(StrictModel):
    """Evaluate every oriented 2-cell boundary in a source-bound complex."""

    complex: FiniteGroupGaugeComplex
    field: FiniteGroupGaugeField


class FiniteGroupGaugeFaceCurvature(StrictModel):
    """The ordered boundary product attached to one oriented face."""

    face_id: GaugeLabel
    value: FiniteGroupTableElement


class FiniteGroupGaugeCurvatureResult(StrictModel):
    """Face holonomies and flatness, bound to their complex and edge field."""

    complex: FiniteGroupGaugeComplex
    field: FiniteGroupGaugeField
    face_values: tuple[FiniteGroupGaugeFaceCurvature, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_FACES
    )
    flat: StrictBool

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        """Bind the rows and the flatness claim to the retained parents.

        The rows are the face holonomies of ``self.complex`` in its own face
        order, and ``flat`` is their own identity claim. Neither is a caller
        statement: a consumer of this carrier reads the face set, the group
        element, and the flatness flag without re-deriving them.
        """
        if (
            self.complex.lattice != self.field.lattice
            or self.complex.group != self.field.group
        ):
            raise _validation_error(
                "curvature_parent",
                "the field must retain the complex lattice and group",
            )
        order = len(self.complex.group.multiplication)
        if tuple(row.face_id for row in self.face_values) != tuple(
            face.face_id for face in self.complex.faces
        ):
            raise _validation_error(
                "curvature_face_binding",
                "curvature rows must cover the retained complex faces in order",
            )
        for row in self.face_values:
            if row.value.group != self.complex.group or row.value.group != (
                self.field.group
            ):
                raise _validation_error(
                    "curvature_parent",
                    "each curvature element must belong to the retained group",
                )
            index = row.value.index
            if type(index) is not int or not 0 <= index < order:
                raise _validation_error(
                    "curvature_element", "curvature elements must be table indices"
                )
        if self.flat != all(
            row.value.index == self.complex.group.identity for row in self.face_values
        ):
            raise _validation_error(
                "curvature_flatness",
                "flatness must be the identity claim of the retained rows",
            )
        return self


class FiniteGroupGaugeHolonomyRequest(StrictModel):
    field: FiniteGroupGaugeField
    path: OrientedGaugePath


class FiniteGroupGaugeContribution(StrictModel):
    edge_id: GaugeLabel
    forward: bool
    value: FiniteGroupTableElement


class FiniteGroupGaugeHolonomyResult(StrictModel):
    field: FiniteGroupGaugeField
    path: OrientedGaugePath
    holonomy: FiniteGroupTableElement
    contributions: tuple[FiniteGroupGaugeContribution, ...] = Field(
        max_length=MAX_GAUGE_PATH_LENGTH
    )
    start: GaugeLabel
    end: GaugeLabel

    @model_validator(mode="after")
    def require_parent_bindings(self) -> Self:
        field = getattr(self, "field", None)
        path = getattr(self, "path", None)
        holonomy = getattr(self, "holonomy", None)
        contributions = getattr(self, "contributions", None)
        group = getattr(field, "group", None)
        table = getattr(group, "multiplication", None)
        order = len(table) if type(table) is tuple else 0
        if (
            not isinstance(field, FiniteGroupGaugeField)
            or not isinstance(getattr(field, "lattice", None), GaugeLattice)
            or not isinstance(group, FiniteGroupTable)
            or type(table) is not tuple
            or not isinstance(path, OrientedGaugePath)
            or not _has_canonical_path_steps(path)
            or not isinstance(holonomy, FiniteGroupTableElement)
            or getattr(holonomy, "group", None) != group
            or type(getattr(holonomy, "index", None)) is not int
            or not 0 <= getattr(holonomy, "index", -1) < order
            or type(contributions) is not tuple
            or len(contributions) != len(path.steps)
        ):
            raise _validation_error(
                "finite_group_holonomy_parent",
                "holonomy result carriers must retain one finite-group field parent",
            )
        for step, contribution in zip(path.steps, contributions, strict=True):
            value = getattr(contribution, "value", None)
            if (
                not isinstance(contribution, FiniteGroupGaugeContribution)
                or getattr(contribution, "edge_id", None) != step.edge_id
                or getattr(contribution, "forward", None) is not step.forward
                or not isinstance(value, FiniteGroupTableElement)
                or getattr(value, "group", None) != group
                or type(getattr(value, "index", None)) is not int
                or not 0 <= getattr(value, "index", -1) < order
            ):
                raise _validation_error(
                    "finite_group_holonomy_contribution_parent",
                    "every path contribution must use the retained field group",
                )
        return self


class FiniteGroupGaugeBasepointTransportRequest(StrictModel):
    """Move a based finite-group loop along an exact lattice path."""

    field: FiniteGroupGaugeField
    loop: OrientedGaugePath
    connector: OrientedGaugePath


class FiniteGroupGaugeBasepointTransportResult(StrictModel):
    """The source loop, connector, and exactly conjugated loop holonomy."""

    field: FiniteGroupGaugeField
    loop: OrientedGaugePath
    connector: OrientedGaugePath
    transported_loop: OrientedGaugePath
    source_basepoint: GaugeLabel
    target_basepoint: GaugeLabel
    source_holonomy: FiniteGroupTableElement
    connector_holonomy: FiniteGroupTableElement
    transported_holonomy: FiniteGroupTableElement

    @model_validator(mode="after")
    def require_parent_binding(self) -> Self:
        if (
            not isinstance(self.field, FiniteGroupGaugeField)
            or not isinstance(self.loop, OrientedGaugePath)
            or not isinstance(self.connector, OrientedGaugePath)
            or not isinstance(self.transported_loop, OrientedGaugePath)
            or self.source_basepoint not in self.field.lattice.vertices
            or self.target_basepoint not in self.field.lattice.vertices
            or self.transported_loop.basepoint != self.target_basepoint
            or any(
                not isinstance(value, FiniteGroupTableElement)
                or value.group != self.field.group
                for value in (
                    self.source_holonomy,
                    self.connector_holonomy,
                    self.transported_holonomy,
                )
            )
        ):
            raise _validation_error(
                "basepoint_transport_parent",
                "transport paths, endpoints, and holonomies must bind to the field",
            )
        self._require_closed_walks()
        return self

    def _require_closed_walks(self) -> None:
        """Every declared path must be a walk of the retained lattice.

        The transported loop is the mathematical claim of this result — it is
        read as a path and its holonomy as a product — so a decoded value must
        not be able to present an open loop, an unrelated connector, or a path
        that never existed on this lattice.
        """
        by_id = {edge.edge_id: edge for edge in self.field.lattice.edges}

        def walk_of(path: OrientedGaugePath) -> tuple[str | None, str | None]:
            try:
                first, last = _lattice_walk_endpoints(
                    by_id, getattr(path, "steps", None)
                )
            except ValueError as exc:
                raise _validation_error("basepoint_transport_walk", str(exc)) from None
            basepoint = getattr(path, "basepoint", None)
            if first is None:
                first = last = basepoint
            if (
                (basepoint is not None and not _is_raw_label(basepoint))
                or first not in self.field.lattice.vertices
                or last not in self.field.lattice.vertices
                or (basepoint is not None and basepoint != first)
            ):
                raise _validation_error(
                    "basepoint_transport_walk",
                    "each path must retain its lattice basepoint",
                )
            return first, last

        loop_first, loop_end = walk_of(self.loop)
        if loop_end != self.source_basepoint or loop_first != self.source_basepoint:
            raise _validation_error(
                "basepoint_transport_loop",
                "the source loop must be closed at the source basepoint",
            )
        connector_first, connector_end = walk_of(self.connector)
        if (
            connector_first != self.source_basepoint
            or connector_end != self.target_basepoint
        ):
            raise _validation_error(
                "basepoint_transport_connector",
                "the connector must join the source and target basepoints",
            )
        if self.transported_loop.steps != (
            tuple(
                GaugePathStep(edge_id=step.edge_id, forward=not step.forward)
                for step in reversed(self.connector.steps)
            )
            + tuple(self.loop.steps)
            + tuple(self.connector.steps)
        ):
            raise _validation_error(
                "basepoint_transport_walk",
                "the transported loop must be the conjugated source walk",
            )
        transported_first, transported_end = walk_of(self.transported_loop)
        if (
            transported_first != self.target_basepoint
            or transported_end != self.target_basepoint
        ):
            raise _validation_error(
                "basepoint_transport_walk",
                "the transported loop must be closed at the target basepoint",
            )


class FiniteGroupGaugeVertexValue(StrictModel):
    """One exact finite-table group element at a lattice vertex."""

    vertex: GaugeLabel
    value: FiniteGroupTableElement


class FiniteGroupGaugeTransformRequest(StrictModel):
    """Apply a complete vertex frame map to one finite-table edge field."""

    field: FiniteGroupGaugeField
    vertex_values: tuple[FiniteGroupGaugeVertexValue, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )


class FiniteGroupGaugeTransformResult(StrictModel):
    """Exact transformed finite-table field with its source and frame map."""

    source: FiniteGroupGaugeField
    transformed: FiniteGroupGaugeField
    vertex_values: tuple[FiniteGroupGaugeVertexValue, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        source = getattr(self, "source", None)
        transformed = getattr(self, "transformed", None)
        vertex_values = getattr(self, "vertex_values", None)
        source_lattice = getattr(source, "lattice", None)
        source_group = getattr(source, "group", None)
        if (
            not isinstance(source, FiniteGroupGaugeField)
            or not isinstance(transformed, FiniteGroupGaugeField)
            or getattr(transformed, "lattice", None) != source_lattice
            or getattr(transformed, "group", None) != source_group
            or type(vertex_values) is not tuple
            or any(
                not isinstance(entry, FiniteGroupGaugeVertexValue)
                for entry in vertex_values
            )
            or tuple(getattr(entry, "vertex", None) for entry in vertex_values)
            != getattr(source_lattice, "vertices", None)
            or any(
                not isinstance(getattr(entry, "value", None), FiniteGroupTableElement)
                or getattr(getattr(entry, "value", None), "group", None) != source_group
                for entry in vertex_values
            )
        ):
            raise _validation_error(
                "finite_group_transform_binding",
                "source, target, and vertex frames must share exact parents",
            )
        return self


class EdgeContribution(StrictModel):
    """The resolved oriented group value of one path step."""

    edge_id: GaugeLabel
    forward: bool
    value: PermutationLabel


class GaugeVertexValue(StrictModel):
    """One exact gauge-frame element at a lattice vertex."""

    vertex: GaugeLabel
    value: PermutationLabel


class GaugeTransformRequest(StrictModel):
    """A finite gauge transform bound to one lattice and group degree."""

    field: GaugeField
    vertex_values: tuple[GaugeVertexValue, ...] = Field(
        min_length=1, max_length=MAX_GAUGE_VERTICES
    )

    @model_validator(mode="after")
    def require_complete_vertex_values(self) -> Self:
        vertices = self.field.lattice.vertices
        labels = {entry.vertex: entry.value for entry in self.vertex_values}
        if set(labels) != set(vertices) or len(labels) != len(self.vertex_values):
            raise _validation_error(
                "transform_vertices",
                "gauge transform must label every lattice vertex exactly once",
            )
        if any(value.degree != self.field.degree for value in labels.values()):
            raise _validation_error(
                "transform_degree", "gauge-frame elements must use the field degree"
            )
        return self


class GaugeTransformResult(StrictModel):
    """The transformed edge field and retained vertex-frame map."""

    source: GaugeField
    transformed: GaugeField
    vertex_values: tuple[GaugeVertexValue, ...]

    @model_validator(mode="after")
    def require_parent_binding(self) -> Self:
        if (
            not isinstance(self.source, GaugeField)
            or not isinstance(self.transformed, GaugeField)
            or not isinstance(self.source.lattice, GaugeLattice)
            or not isinstance(self.transformed.lattice, GaugeLattice)
        ):
            raise _validation_error(
                "transform_parent", "source and transformed values must be gauge fields"
            )
        if self.source.lattice != self.transformed.lattice:
            raise _validation_error(
                "transform_parent",
                "source and transformed fields must share one lattice",
            )
        if self.source.degree != self.transformed.degree:
            raise _validation_error(
                "transform_degree",
                "source and transformed fields must share one group degree",
            )
        vertices = self.source.lattice.vertices
        if not isinstance(self.vertex_values, tuple) or any(
            not isinstance(entry, GaugeVertexValue) for entry in self.vertex_values
        ):
            raise _validation_error(
                "transform_vertices", "vertex frames must be typed gauge values"
            )
        if any(
            not isinstance(entry.value, PermutationLabel)
            or type(entry.value.degree) is not int
            or not isinstance(entry.value.image, tuple)
            or len(entry.value.image) != entry.value.degree
            or any(type(value) is not int for value in entry.value.image)
            or sorted(entry.value.image) != list(range(entry.value.degree))
            for entry in self.vertex_values
        ):
            raise _validation_error(
                "transform_vertices", "vertex frames must be valid permutation values"
            )
        labels = tuple(entry.vertex for entry in self.vertex_values)
        if tuple(sorted(labels)) != tuple(sorted(vertices)) or len(set(labels)) != len(
            labels
        ):
            raise _validation_error(
                "transform_vertices",
                "vertex frames must cover the source lattice exactly once",
            )
        if any(
            entry.value.degree != self.source.degree for entry in self.vertex_values
        ):
            raise _validation_error(
                "transform_degree",
                "every vertex frame must use the source group degree",
            )
        return self


class PlaquetteRequest(StrictModel):
    """A closed oriented lattice path whose holonomy is curvature."""

    field: GaugeField
    path: OrientedGaugePath


class PlaquetteResult(StrictModel):
    """Exact oriented plaquette curvature bound to field and path."""

    field: GaugeField
    path: OrientedGaugePath
    curvature: PermutationLabel
    start: GaugeLabel

    @model_validator(mode="after")
    def require_path_and_field_binding(self) -> Self:
        if (
            not isinstance(self.field, GaugeField)
            or not isinstance(self.field.lattice, GaugeLattice)
            or not isinstance(self.path, OrientedGaugePath)
            or not isinstance(self.curvature, PermutationLabel)
            or type(self.field.degree) is not int
            or type(self.curvature.degree) is not int
            or any(not isinstance(edge, GaugeEdge) for edge in self.field.lattice.edges)
        ):
            raise _validation_error(
                "plaquette_parent", "plaquette source and path are malformed"
            )
        if (
            not isinstance(self.curvature.image, tuple)
            or len(self.curvature.image) != self.curvature.degree
            or any(type(value) is not int for value in self.curvature.image)
            or sorted(self.curvature.image) != list(range(self.curvature.degree))
        ):
            raise _validation_error(
                "plaquette_parent", "plaquette curvature is not a permutation value"
            )
        if self.curvature.degree != self.field.degree:
            raise _validation_error(
                "plaquette_degree", "curvature must use the field's group degree"
            )
        by_id = {edge.edge_id: edge for edge in self.field.lattice.edges}
        if not isinstance(self.path.steps, tuple) or not self.path.steps:
            raise _validation_error("plaquette_path", "plaquette path must be nonempty")
        cursor: str | None = None
        first: str | None = None
        for step in self.path.steps:
            if not isinstance(step, GaugePathStep) or type(step.forward) is not bool:
                raise _validation_error(
                    "plaquette_path", "plaquette steps must be typed lattice traversals"
                )
            edge = by_id.get(step.edge_id)
            if edge is None:
                raise _validation_error(
                    "plaquette_path", "plaquette path must use source-lattice edges"
                )
            tail, head = (
                (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
            )
            if cursor is not None and tail != cursor:
                raise _validation_error(
                    "plaquette_path", "plaquette steps must chain head-to-tail"
                )
            if first is None:
                first = tail
            cursor = head
        if first != cursor or self.start != first:
            raise _validation_error(
                "plaquette_path",
                "plaquette start must bind to a closed source-lattice path",
            )
        return self


class HolonomyRequest(StrictModel):
    """Compute the ordered exact group product along an oriented edge path.

    Path steps must chain head-to-tail over the field's lattice, and every
    step must reference a labelled lattice edge. Reverse traversal resolves
    to the exact group inverse of the forward label.
    """

    field: GaugeField
    path: OrientedGaugePath


class GaugeLoopFamilyRequest(StrictModel):
    """Evaluate an explicit finite family of loops over one permutation field."""

    field: GaugeField
    loops: tuple[OrientedGaugePath, ...] = Field(
        max_length=MAX_GAUGE_LOOP_FAMILY_SIZE,
        description=(
            f"At most {MAX_GAUGE_LOOP_FAMILY_SIZE} loops, each with at most "
            f"{MAX_GAUGE_PATH_LENGTH} steps and at most "
            f"{MAX_GAUGE_LOOP_FAMILY_STEPS} steps total across the family."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_family_steps(self) -> Self:
        """Bound the aggregate step count across the admitted loop family.

        The per-loop count and each path's step count are already bounded by
        their field declarations, so only the aggregate remains. Validating
        after parsing keeps this model on the strict JSON contract: a
        ``mode="before"`` validator makes Pydantic validate the whole model
        against Python objects, which rejects JSON arrays for the
        tuple-typed field, vertex, and edge axes.
        """
        if sum(len(loop.steps) for loop in self.loops) > MAX_GAUGE_LOOP_FAMILY_STEPS:
            raise _validation_error(
                "loop_family_steps",
                "aggregate loop-family paths may contain at most "
                f"{MAX_GAUGE_LOOP_FAMILY_STEPS} steps",
            )
        return self


class GaugeLoopHolonomy(StrictModel):
    """One based loop and its exact holonomy in a family result."""

    path: OrientedGaugePath
    basepoint: GaugeLabel
    holonomy: PermutationLabel


def _loop_family_output_units(
    field: GaugeField, loops: tuple[GaugeLoopHolonomy, ...]
) -> int:
    degree = field.degree
    units = 32
    units += sum(len(vertex) + 4 for vertex in field.lattice.vertices)
    units += sum(
        len(edge.edge_id) + len(edge.tail) + len(edge.head) + 8
        for edge in field.lattice.edges
    )
    units += sum(len(entry.edge_id) + degree + 4 for entry in field.edge_labels)
    for entry in loops:
        units += len(entry.basepoint) + degree + 12
        units += sum(len(step.edge_id) + 4 for step in entry.path.steps)
    return units


def _check_raw_permutation_shape(value: object) -> None:
    fields = _raw_fields(
        value, (PermutationLabel,), {"degree", "image"}, "loop_family_parent"
    )
    degree = fields.get("degree")
    if type(degree) is not int or not MIN_GAUGE_DEGREE <= degree <= MAX_GAUGE_DEGREE:
        raise ValueError("permutation degree must be bounded")
    image = _raw_sequence(fields.get("image"), MAX_GAUGE_DEGREE, "loop_family_parent")
    if len(image) != degree or any(
        type(entry) is not int or not 0 <= entry < degree for entry in image
    ):
        raise ValueError("permutation images must contain bounded scalar indices")


def _preflight_loop_family(value: object) -> None:
    """Bound native and raw fields before nested validation, counting, or copies."""
    try:
        family = _raw_fields(
            value,
            (GaugeLoopFamilyHolonomies,),
            {"field", "loops"},
            "loop_family_parent",
        )
        field = _raw_fields(
            family.get("field"),
            (GaugeField,),
            {"lattice", "degree", "edge_labels"},
            "loop_family_parent",
        )
        _check_raw_lattice_shape(field.get("lattice"))
        degree = field.get("degree")
        if (
            type(degree) is not int
            or not MIN_GAUGE_DEGREE <= degree <= MAX_GAUGE_DEGREE
        ):
            raise ValueError("field degree must be bounded")
        labels = _raw_sequence(
            field.get("edge_labels"), MAX_GAUGE_EDGES, "loop_family_parent"
        )
        for label in labels:
            fields = _raw_fields(
                label,
                (GaugeFieldEdgeLabel,),
                {"edge_id", "label"},
                "loop_family_parent",
            )
            if not _is_raw_label(fields.get("edge_id")):
                raise ValueError("field edge IDs must be bounded scalar labels")
            _check_raw_permutation_shape(fields.get("label"))
        loops = _raw_sequence(
            family.get("loops"), MAX_GAUGE_LOOP_FAMILY_SIZE, "loop_family_parent"
        )
        total_steps = 0
        for entry in loops:
            fields = _raw_fields(
                entry,
                (GaugeLoopHolonomy,),
                {"path", "basepoint", "holonomy"},
                "loop_family_parent",
            )
            if not _is_raw_label(fields.get("basepoint")):
                raise ValueError("loop basepoints must be bounded scalar labels")
            total_steps += _raw_path_step_count(fields.get("path"))
            _check_raw_permutation_shape(fields.get("holonomy"))
    except (AttributeError, TypeError, ValueError):
        raise _validation_error(
            "loop_family_parent",
            "loop-family source and entries must be bounded canonical carriers",
        ) from None
    if total_steps > MAX_GAUGE_LOOP_FAMILY_STEPS:
        raise _validation_error(
            "loop_family_steps",
            "aggregate loop-family paths may contain at most 4096 steps",
        )


class GaugeLoopFamilyHolonomies(StrictModel):
    """Holonomies of explicit loops, bound to one source field exactly once."""

    field: GaugeField
    loops: tuple[GaugeLoopHolonomy, ...] = Field(
        max_length=MAX_GAUGE_LOOP_FAMILY_SIZE,
        description=(
            f"At most {MAX_GAUGE_LOOP_FAMILY_SIZE} loops, each with at most "
            f"{MAX_GAUGE_PATH_LENGTH} steps and at most "
            f"{MAX_GAUGE_LOOP_FAMILY_STEPS} steps total across the family."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_nested_carriers(cls, value: object) -> object:
        _preflight_loop_family(value)
        return canonicalize_json_containers(value)

    @model_validator(mode="after")
    def require_source_bound_closed_loops(self) -> Self:
        # Existing model instances may skip the before-validator. Establish the
        # same bounded shape facts before traversing or dumping their children.
        _preflight_loop_family(self)
        try:
            field = GaugeField.model_validate(self.field.model_dump())
            loops = tuple(
                GaugeLoopHolonomy.model_validate(entry.model_dump())
                for entry in self.loops
            )
            if field != self.field or loops != self.loops:
                raise ValueError("loop-family carriers are not canonical")
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise _validation_error(
                "loop_family_parent", "loop-family source is malformed"
            ) from None
        by_id = {edge.edge_id: edge for edge in field.lattice.edges}
        for entry in loops:
            path = entry.path
            if entry.holonomy.degree != field.degree:
                raise _validation_error(
                    "loop_family_holonomy",
                    "every loop holonomy must belong to the source permutation group",
                )
            loop_first, loop_last = _lattice_walk_endpoints(by_id, path.steps)
            if loop_first is None:
                loop_first = loop_last = path.basepoint
            if (
                loop_first not in field.lattice.vertices
                or loop_first != loop_last
                or entry.basepoint != loop_first
                or (path.basepoint is not None and path.basepoint != loop_first)
            ):
                raise _validation_error(
                    "loop_family_path",
                    "each loop must be a closed walk based at its declared basepoint",
                )
        if _loop_family_output_units(field, loops) > MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS:
            raise _validation_error(
                "loop_family_output",
                "source-bound loop family exceeds its exact output envelope",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, field: GaugeField, loops: tuple[GaugeLoopHolonomy, ...]
    ) -> Self:
        """Build a trusted family outcome without replaying its products."""

        return cls.model_construct(field=field, loops=loops)


class PermutationWilsonTraceRequest(StrictModel):
    """Evaluate the natural permutation-character Wilson loop over ``S_d``."""

    field: GaugeField
    path: OrientedGaugePath


class PermutationWilsonTraceResult(StrictModel):
    """Exact trace in the natural degree-``d`` permutation representation."""

    field: GaugeField
    path: OrientedGaugePath
    holonomy: PermutationLabel
    trace: StrictInt = Field(ge=0, le=MAX_GAUGE_DEGREE)


class HolonomyResult(StrictModel):
    """Ordered holonomy bound to its source field and oriented path."""

    field: GaugeField
    path: OrientedGaugePath

    holonomy: PermutationLabel
    contributions: tuple[EdgeContribution, ...] = Field(
        max_length=MAX_GAUGE_PATH_LENGTH
    )
    start: GaugeLabel
    end: GaugeLabel

    @model_validator(mode="after")
    def require_holonomy_shape(self) -> Self:
        if any(
            contribution.value.degree != self.holonomy.degree
            for contribution in self.contributions
        ):
            raise _validation_error(
                "holonomy_degree",
                "every contribution must use the holonomy structure degree",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        field: GaugeField,
        path: OrientedGaugePath,
        holonomy: PermutationLabel,
        contributions: tuple[EdgeContribution, ...],
        start: str,
        end: str,
    ) -> Self:
        """Build a trusted kernel outcome without replaying its product."""

        return cls.model_construct(
            field=field,
            path=path,
            holonomy=holonomy,
            contributions=contributions,
            start=start,
            end=end,
        )


__all__ = [
    "MAX_FINITE_GROUP_GAUGE_COMPLEX_OUTPUT_UNITS",
    "MAX_GAUGE_DEGREE",
    "MAX_GAUGE_EDGES",
    "MAX_GAUGE_FACES",
    "MAX_GAUGE_LABEL_LENGTH",
    "MAX_GAUGE_LOOP_FAMILY_OUTPUT_UNITS",
    "MAX_GAUGE_LOOP_FAMILY_SIZE",
    "MAX_GAUGE_LOOP_FAMILY_STEPS",
    "MAX_GAUGE_LOOP_FAMILY_WORK",
    "MAX_GAUGE_PATH_LENGTH",
    "MAX_GAUGE_TOTAL_FACE_STEPS",
    "MAX_GAUGE_VERTICES",
    "MIN_GAUGE_DEGREE",
    "EdgeContribution",
    "FiniteGroupGaugeBasepointTransportRequest",
    "FiniteGroupGaugeBasepointTransportResult",
    "FiniteGroupGaugeComplex",
    "FiniteGroupGaugeComplexRequest",
    "FiniteGroupGaugeFace",
    "GaugeEdge",
    "GaugeField",
    "GaugeFieldEdgeLabel",
    "GaugeLabel",
    "GaugeLattice",
    "GaugeLoopFamilyHolonomies",
    "GaugeLoopFamilyRequest",
    "GaugeLoopHolonomy",
    "GaugePathStep",
    "GaugeTransformRequest",
    "GaugeTransformResult",
    "GaugeVertexValue",
    "HolonomyRequest",
    "HolonomyResult",
    "OrientedGaugePath",
    "PermutationLabel",
    "PlaquetteRequest",
    "PlaquetteResult",
]
