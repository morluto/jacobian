"""Source-bound finite-gauge 2-complex construction."""

from __future__ import annotations

from typing import NoReturn, cast

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.gauge._models import (
    MAX_GAUGE_EDGES,
    MAX_GAUGE_FACES,
    MAX_GAUGE_TOTAL_FACE_STEPS,
    MAX_GAUGE_VERTICES,
    FiniteGroupGaugeComplex,
    FiniteGroupGaugeComplexRequest,
    FiniteGroupGaugeFace,
    GaugeEdge,
    GaugeLattice,
    GaugePathStep,
    OrientedGaugePath,
)
from jacobian.math.gauge.finite_group import _admit_group
from jacobian.math.groups._table_models import FiniteGroupTable


def _reject(location: str, code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=(location,), code=code, message=message
    )


def _is_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 64
        and not any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    )


def _admit_lattice(
    lattice: GaugeLattice,
) -> tuple[tuple[str, ...], tuple[GaugeEdge, ...], set[str], dict[str, GaugeEdge]]:
    vertices = getattr(lattice, "vertices", None)
    edges = getattr(lattice, "edges", None)
    if (
        not isinstance(vertices, tuple)
        or not 1 <= len(vertices) <= MAX_GAUGE_VERTICES
        or any(not _is_label(value) for value in vertices)
        or len(set(vertices)) != len(vertices)
    ):
        _reject(
            "lattice", "lattice_gauge.complex.lattice_shape", "vertices are malformed"
        )
    if (
        not isinstance(edges, tuple)
        or not 1 <= len(edges) <= MAX_GAUGE_EDGES
        or any(not isinstance(edge, GaugeEdge) for edge in edges)
    ):
        _reject("lattice", "lattice_gauge.complex.lattice_shape", "edges are malformed")
    edge_ids: list[str] = []
    vertex_set = set(vertices)
    for edge in edges:
        edge_id = getattr(edge, "edge_id", None)
        tail = getattr(edge, "tail", None)
        head = getattr(edge, "head", None)
        if any(not _is_label(value) for value in (edge_id, tail, head)):
            _reject(
                "lattice", "lattice_gauge.complex.lattice_shape", "edge is malformed"
            )
        if tail not in vertex_set or head not in vertex_set:
            _reject(
                "lattice", "lattice_gauge.complex.lattice_shape", "edge is malformed"
            )
        edge_ids.append(cast(str, edge_id))
    if tuple(sorted(edge_ids)) != tuple(edge_ids) or len(set(edge_ids)) != len(
        edge_ids
    ):
        _reject(
            "lattice",
            "lattice_gauge.complex.edge_ids",
            "edge IDs must be unique and ordered",
        )
    edge_by_id = {edge.edge_id: edge for edge in edges}
    return vertices, edges, vertex_set, edge_by_id


def _admit_faces(
    faces: object,
    vertex_set: set[str],
    edge_by_id: dict[str, GaugeEdge],
) -> None:
    if (
        not isinstance(faces, tuple)
        or len(faces) > MAX_GAUGE_FACES
        or any(not isinstance(face, FiniteGroupGaugeFace) for face in faces)
    ):
        _reject("faces", "lattice_gauge.complex.face_shape", "faces are malformed")
    face_ids: list[str] = []
    total_steps = 0
    for face in faces:
        face_id = getattr(face, "face_id", None)
        path = getattr(face, "boundary", None)
        if not _is_label(face_id) or not isinstance(path, OrientedGaugePath):
            _reject("faces", "lattice_gauge.complex.face_shape", "face is malformed")
        face_ids.append(cast(str, face_id))
        steps = getattr(path, "steps", None)
        basepoint = getattr(path, "basepoint", None)
        if not isinstance(steps, tuple) or len(steps) > 256:
            raise OperationResourceAdmissionError(
                location=("faces", cast(str, face_id)),
                code="lattice_gauge.complex.face_length",
                message="a face boundary may contain at most 256 oriented edge steps",
            )
        total_steps += len(steps)
        if total_steps > MAX_GAUGE_TOTAL_FACE_STEPS:
            raise OperationResourceAdmissionError(
                location=("faces",),
                code="lattice_gauge.complex.total_face_steps",
                message="aggregate face boundary exceeds 4096 oriented edge steps",
            )
        _admit_one_face(
            steps,
            basepoint,
            vertex_set,
            edge_by_id,
        )
    if tuple(sorted(face_ids)) != tuple(face_ids) or len(set(face_ids)) != len(
        face_ids
    ):
        _reject(
            "faces",
            "lattice_gauge.complex.face_ids",
            "face IDs must be unique and ordered",
        )


def _admit_one_face(
    steps: tuple[GaugePathStep, ...],
    basepoint: object,
    vertex_set: set[str],
    edge_by_id: dict[str, GaugeEdge],
    total_steps: int,
 ) -> None:
    if not steps:
        if not _is_label(basepoint) or basepoint not in vertex_set:
            _reject(
                "faces",
                "lattice_gauge.complex.empty_face_basepoint",
                "constant face attachment must name a source lattice vertex",
            )
        return
    first: str | None = None
    cursor: str | None = None
    for step in steps:
        if (
            not isinstance(step, GaugePathStep)
            or type(getattr(step, "forward", None)) is not bool
            or not _is_label(getattr(step, "edge_id", None))
        ):
            _reject(
                "faces", "lattice_gauge.complex.step_shape", "face step is malformed"
            )
        edge_id = step.edge_id
        edge = edge_by_id.get(edge_id)
        if edge is None:
            _reject(
                "faces",
                "lattice_gauge.complex.face_edge",
                "face must use source lattice edges",
            )
        tail, head = (edge.tail, edge.head) if step.forward else (edge.head, edge.tail)
        if cursor is not None and cursor != tail:
            _reject(
                "faces",
                "lattice_gauge.complex.face_chain",
                "face steps must chain head-to-tail",
            )
        if first is None:
            first = tail
        cursor = head
    if first != cursor or (basepoint is not None and basepoint != first):
        _reject(
            "faces",
            "lattice_gauge.complex.face_closed",
            "each oriented face boundary must be closed",
        )


def _admit(request: FiniteGroupGaugeComplexRequest) -> None:
    """Validate source identities and bound work/output before result creation."""
    if not isinstance(request, FiniteGroupGaugeComplexRequest):
        _reject(
            "request", "lattice_gauge.complex.request_shape", "request is malformed"
        )
    lattice = getattr(request, "lattice", None)
    group = getattr(request, "group", None)
    faces = getattr(request, "faces", None)
    if not isinstance(group, FiniteGroupTable):
        _reject(
            "group", "lattice_gauge.complex.group_shape", "group table is malformed"
        )
    if not isinstance(lattice, GaugeLattice):
        _reject(
            "lattice", "lattice_gauge.complex.lattice_shape", "lattice is malformed"
        )
    try:
        table, inverse, identity, _order = _admit_group(group, location="group")
    except (AttributeError, TypeError):
        _reject(
            "group", "lattice_gauge.complex.group_shape", "group table is malformed"
        )
    del table, inverse, identity
    _, _, vertex_set, edge_by_id, _ = _admit_lattice(lattice)
    _admit_faces(faces, vertex_set, edge_by_id)


def construct_finite_group_gauge_complex(
    request: FiniteGroupGaugeComplexRequest,
) -> FiniteGroupGaugeComplex:
    """Admit and retain a finite lattice with ordered oriented face boundaries."""
    _admit(request)
    return FiniteGroupGaugeComplex.model_construct(
        lattice=request.lattice,
        group=request.group,
        faces=request.faces,
    )


__all__ = ["construct_finite_group_gauge_complex"]
