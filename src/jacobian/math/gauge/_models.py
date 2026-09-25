"""Typed contracts for exact permutation-valued lattice-gauge holonomy."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.groups._table_models import (
    MAX_FINITE_TABLE_GROUP_ORDER,
    FiniteGroupTable,
    FiniteGroupTableElement,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"lattice_gauge.{reason}", message)


def _json_arrays_to_tuples(value: object) -> object:
    """Decode only declared JSON array fields as tuples for strict round trips."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    lattice = result.get("lattice")
    if isinstance(lattice, dict):
        lattice = dict(lattice)
        for key in ("vertices", "edges"):
            if isinstance(lattice.get(key), (list, tuple)):
                lattice[key] = tuple(lattice[key])
        result["lattice"] = lattice
    group = result.get("group")
    if isinstance(group, dict):
        group = dict(group)
        multiplication = group.get("multiplication")
        if isinstance(multiplication, (list, tuple)):
            group["multiplication"] = tuple(
                tuple(row) if isinstance(row, (list, tuple)) else row
                for row in multiplication
            )
        if isinstance(group.get("inverse"), (list, tuple)):
            group["inverse"] = tuple(group["inverse"])
        result["group"] = group
    faces = result.get("faces")
    if isinstance(faces, (list, tuple)):
        normalized_faces = []
        for face in faces:
            if isinstance(face, dict):
                face = dict(face)
                boundary = face.get("boundary")
                if isinstance(boundary, dict):
                    boundary = dict(boundary)
                    if isinstance(boundary.get("steps"), (list, tuple)):
                        boundary["steps"] = tuple(boundary["steps"])
                    face["boundary"] = boundary
            normalized_faces.append(face)
        result["faces"] = tuple(normalized_faces)
    return result


def _check_raw_complex_shape(value: object) -> None:
    """Bound every public array before copying JSON lists into canonical tuples."""
    if not isinstance(value, dict):
        return
    if set(value) - {"lattice", "group", "faces"}:
        raise _validation_error(
            "complex_request_shape", "complex request has unknown fields"
        )
    _check_raw_lattice_shape(value.get("lattice"))
    _check_raw_group_shape(value.get("group"))
    _check_raw_faces_shape(value.get("faces"))


def _check_raw_lattice_shape(lattice: object) -> None:
    if isinstance(lattice, dict):
        if set(lattice) - {"vertices", "edges"}:
            raise _validation_error(
                "complex_lattice_shape", "lattice has unknown fields"
            )
        vertices = lattice.get("vertices")
        edges = lattice.get("edges")
        if isinstance(vertices, (tuple, list)) and len(vertices) > MAX_GAUGE_VERTICES:
            raise _validation_error(
                "complex_vertex_bound", "lattice exceeds the vertex bound"
            )
        if isinstance(edges, (tuple, list)) and len(edges) > MAX_GAUGE_EDGES:
            raise _validation_error(
                "complex_edge_bound", "lattice exceeds the edge bound"
            )


def _check_raw_group_shape(group: object) -> None:
    if isinstance(group, dict):
        if set(group) - {"multiplication", "identity", "inverse"}:
            raise _validation_error(
                "complex_group_shape", "group table has unknown fields"
            )
        multiplication = group.get("multiplication")
        inverse = group.get("inverse")
        if isinstance(multiplication, (tuple, list)) and (
            len(multiplication) > MAX_FINITE_TABLE_GROUP_ORDER
            or any(
                isinstance(row, (tuple, list))
                and len(row) > MAX_FINITE_TABLE_GROUP_ORDER
                for row in multiplication
            )
        ):
            raise _validation_error(
                "complex_group_bound", "group table exceeds order 24"
            )
        if (
            isinstance(inverse, (tuple, list))
            and len(inverse) > MAX_FINITE_TABLE_GROUP_ORDER
        ):
            raise _validation_error(
                "complex_group_bound", "group table exceeds order 24"
            )


def _check_raw_faces_shape(faces: object) -> None:
    if not isinstance(faces, (tuple, list)):
        return
    if len(faces) > MAX_GAUGE_FACES:
        raise _validation_error("complex_face_bound", "complex exceeds the face bound")
    total_steps = 0
    for face in faces:
        total_steps += _raw_face_step_count(face)
        if total_steps > MAX_GAUGE_TOTAL_FACE_STEPS:
            raise _validation_error(
                "complex_boundary_bound",
                "aggregate face boundary exceeds the finite complex envelope",
            )


def _raw_face_step_count(face: object) -> int:
    if not isinstance(face, dict):
        return 0
    if set(face) - {"face_id", "boundary"}:
        raise _validation_error("complex_face_shape", "face has unknown fields")
    boundary = face.get("boundary")
    if not isinstance(boundary, dict):
        return 0
    if set(boundary) - {"steps", "basepoint"}:
        raise _validation_error("complex_boundary_shape", "boundary has unknown fields")
    steps = boundary.get("steps")
    if not isinstance(steps, (tuple, list)):
        return 0
    if len(steps) > MAX_GAUGE_PATH_LENGTH:
        raise _validation_error(
            "complex_face_length", "face exceeds 256 boundary steps"
        )
    for step in steps:
        if isinstance(step, dict) and set(step) - {"edge_id", "forward"}:
            raise _validation_error(
                "complex_step_shape", "face step has unknown fields"
            )
    return len(steps)


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

"""Maximum conservative serialized size of one finite gauge complex."""

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
        if not self.steps and self.basepoint is None:
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
        order = len(self.group.multiplication)
        if any(
            not isinstance(value.value, FiniteGroupTableElement)
            or value.value.group != self.group
            or value.value.index >= order
            for value in self.edge_values
        ):
            raise _validation_error(
                "finite_group_value_parent",
                "every edge element must use the field's exact table parent",
            )
        have = tuple(value.edge_id for value in self.edge_values)
        want = tuple(edge.edge_id for edge in self.lattice.edges)
        if tuple(sorted(have)) != tuple(sorted(set(have))) or set(have) != set(want):
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
    faces: tuple[FiniteGroupGaugeFace, ...] = Field(max_length=MAX_GAUGE_FACES)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_json_arrays(cls, value: object) -> object:
        _check_raw_complex_shape(value)
        value = _json_arrays_to_tuples(value)
        return value

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
        return self


class FiniteGroupGaugeComplexRequest(StrictModel):
    """Construct source-bound oriented face boundaries over one gauge lattice."""

    lattice: GaugeLattice
    group: FiniteGroupTable
    faces: tuple[FiniteGroupGaugeFace, ...] = Field(max_length=MAX_GAUGE_FACES)

    @model_validator(mode="before")
    @classmethod
    def admit_raw_face_growth(cls, value: object) -> object:
        _check_raw_complex_shape(value)
        return _json_arrays_to_tuples(value)


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
    "MAX_GAUGE_DEGREE",
    "MAX_GAUGE_EDGES",
    "MAX_GAUGE_FACES",
    "MAX_GAUGE_LABEL_LENGTH",
    "MAX_GAUGE_PATH_LENGTH",
    "MAX_GAUGE_TOTAL_FACE_STEPS",
    "MAX_GAUGE_VERTICES",
    "MIN_GAUGE_DEGREE",
    "EdgeContribution",
    "FiniteGroupGaugeComplex",
    "FiniteGroupGaugeComplexRequest",
    "FiniteGroupGaugeFace",
    "GaugeEdge",
    "GaugeField",
    "GaugeFieldEdgeLabel",
    "GaugeLabel",
    "GaugeLattice",
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
