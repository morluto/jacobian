"""Domain adapter for combinatorial-map operations."""

from __future__ import annotations

from jacobian.math.topology.combinatorial_maps._models import (
    ConnectedComponentsRequest,
    ConnectedComponentsResult,
    DualRequest,
    DualResult,
    EulerCharacteristicRequest,
    EulerCharacteristicResult,
    FacesRequest,
    FacesResult,
    OrientableGenusRequest,
    OrientableGenusResult,
    OrientationReverseRequest,
    OrientationReverseResult,
    VertexFaceIncidenceRequest,
    VertexFaceIncidenceResult,
)
from jacobian.math.topology.combinatorial_maps.operations import (
    connected_components,
    dual_map,
    euler_characteristic,
    face_orbits,
    orientable_genus,
    orientation_reverse,
    vertex_face_incidence,
)

__all__ = [
    "compute_connected_components",
    "compute_dual",
    "compute_euler_characteristic",
    "compute_faces",
    "compute_orientable_genus",
    "compute_orientation_reverse",
    "compute_vertex_face_incidence",
]


def compute_faces(request: FacesRequest) -> FacesResult:
    return face_orbits(request.map)


def compute_euler_characteristic(
    request: EulerCharacteristicRequest,
) -> EulerCharacteristicResult:
    return euler_characteristic(request.map)


def compute_orientable_genus(
    request: OrientableGenusRequest,
) -> OrientableGenusResult:
    return orientable_genus(request.map)


def compute_orientation_reverse(
    request: OrientationReverseRequest,
) -> OrientationReverseResult:
    return orientation_reverse(request.map)


def compute_connected_components(
    request: ConnectedComponentsRequest,
) -> ConnectedComponentsResult:
    return connected_components(request.map)


def compute_dual(request: DualRequest) -> DualResult:
    return dual_map(request.map)


def compute_vertex_face_incidence(
    request: VertexFaceIncidenceRequest,
) -> VertexFaceIncidenceResult:
    return vertex_face_incidence(request.map)
