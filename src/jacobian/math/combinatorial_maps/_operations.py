"""Domain adapter for combinatorial-map operations."""

from __future__ import annotations

from jacobian.math.combinatorial_maps._models import (
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
from jacobian.math.combinatorial_maps.operations import (
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
    "verify_connected_components_result",
    "verify_dual_result",
    "verify_euler_characteristic_result",
    "verify_faces_result",
    "verify_orientable_genus_result",
    "verify_orientation_reverse_result",
    "verify_vertex_face_incidence_result",
]


def compute_faces(request: FacesRequest) -> FacesResult:
    walks, face_of_dart, successor, _ = face_orbits(request.map)
    n = len(request.map.darts)
    return FacesResult._from_kernel(
        request,
        face_walks=tuple(tuple(walk) for walk in walks),
        face_of_dart=tuple(face_of_dart[d] for d in range(n)),
        successor=tuple(successor),
    )


def compute_euler_characteristic(
    request: EulerCharacteristicRequest,
) -> EulerCharacteristicResult:
    per_component, total = euler_characteristic(request.map)
    return EulerCharacteristicResult._from_kernel(
        per_component=tuple(
            {"V": row["V"], "E": row["E"], "F": row["F"], "chi": row["chi"]}
            for row in per_component
        ),
        total={
            "V": total["V"],
            "E": total["E"],
            "F": total["F"],
            "chi": total["chi"],
        },
    )


def compute_orientable_genus(
    request: OrientableGenusRequest,
) -> OrientableGenusResult:
    per_component, total = orientable_genus(request.map)
    return OrientableGenusResult._from_kernel(
        per_component=tuple(per_component),
        total=total,
    )


def compute_orientation_reverse(
    request: OrientationReverseRequest,
) -> OrientationReverseResult:
    reversed_map, bijection = orientation_reverse(request.map)
    return OrientationReverseResult._from_kernel(
        request,
        reversed_map=reversed_map,
        face_bijection=bijection,
    )


def compute_connected_components(
    request: ConnectedComponentsRequest,
) -> ConnectedComponentsResult:
    vertex_component, dart_component, face_component = connected_components(request.map)
    n_vertices = request.map.vertex_count
    n_darts = len(request.map.darts)
    walks, _, _, _ = face_orbits(request.map)
    n_faces = len(walks)
    return ConnectedComponentsResult._from_kernel(
        vertex_component=tuple(vertex_component[v] for v in range(n_vertices)),
        dart_component=tuple(dart_component[d] for d in range(n_darts)),
        face_component=tuple(face_component[f] for f in range(n_faces)),
    )


def compute_dual(request: DualRequest) -> DualResult:
    dual, primal_to_dual = dual_map(request.map)
    return DualResult._from_kernel(dual=dual, primal_to_dual=primal_to_dual)


def compute_vertex_face_incidence(
    request: VertexFaceIncidenceRequest,
) -> VertexFaceIncidenceResult:
    multiplicity, boolean = vertex_face_incidence(request.map)
    nested: dict[int, dict[int, int]] = {}
    for (vertex, face), count in multiplicity.items():
        nested.setdefault(vertex, {})[face] = count
    boolean_incidence = {v: tuple(sorted(boolean[v])) for v in sorted(boolean)}
    return VertexFaceIncidenceResult._from_kernel(
        multiplicity=nested,
        boolean_incidence=boolean_incidence,
    )


def verify_faces_result(result: FacesResult) -> bool:
    """Verify one source-bound face result inside the admitted map envelope."""

    walks, face_of_dart, successor, _ = face_orbits(result.map)
    dart_count = len(result.map.darts)
    return (
        result.face_walks == tuple(tuple(walk) for walk in walks)
        and result.face_of_dart == tuple(face_of_dart[d] for d in range(dart_count))
        and result.successor == tuple(successor)
    )


def verify_euler_characteristic_result(
    request: EulerCharacteristicRequest, result: EulerCharacteristicResult
) -> bool:
    """Verify one Euler-characteristic claim in the admitted map envelope."""

    per_component, total = euler_characteristic(request.map)
    expected_components = tuple(
        {"V": row["V"], "E": row["E"], "F": row["F"], "chi": row["chi"]}
        for row in per_component
    )
    expected_total = {
        "V": total["V"],
        "E": total["E"],
        "F": total["F"],
        "chi": total["chi"],
    }
    return (
        result.per_component == expected_components and result.total == expected_total
    )


def verify_orientable_genus_result(
    request: OrientableGenusRequest, result: OrientableGenusResult
) -> bool:
    """Verify one orientable-genus claim in the admitted map envelope."""

    per_component, total = orientable_genus(request.map)
    return result.per_component == tuple(per_component) and result.total == total


def verify_orientation_reverse_result(result: OrientationReverseResult) -> bool:
    """Verify one source-bound orientation-reversal claim."""

    reversed_map, face_bijection = orientation_reverse(result.map)
    return (
        result.reversed_map == reversed_map and result.face_bijection == face_bijection
    )


def verify_connected_components_result(
    request: ConnectedComponentsRequest, result: ConnectedComponentsResult
) -> bool:
    """Verify one component-partition claim in the admitted map envelope."""

    vertex_component, dart_component, face_component = connected_components(request.map)
    walks, _, _, _ = face_orbits(request.map)
    return (
        result.vertex_component
        == tuple(vertex_component[v] for v in range(request.map.vertex_count))
        and result.dart_component
        == tuple(dart_component[d] for d in range(len(request.map.darts)))
        and result.face_component == tuple(face_component[f] for f in range(len(walks)))
    )


def verify_dual_result(request: DualRequest, result: DualResult) -> bool:
    """Verify one dual-map claim in the admitted map envelope."""

    dual, primal_to_dual = dual_map(request.map)
    return result.dual == dual and result.primal_to_dual == primal_to_dual


def verify_vertex_face_incidence_result(
    request: VertexFaceIncidenceRequest, result: VertexFaceIncidenceResult
) -> bool:
    """Verify one vertex--face incidence claim in the admitted map envelope."""

    multiplicity, boolean = vertex_face_incidence(request.map)
    nested: dict[int, dict[int, int]] = {}
    for (vertex, face), count in multiplicity.items():
        nested.setdefault(vertex, {})[face] = count
    expected_boolean = {v: tuple(sorted(boolean[v])) for v in sorted(boolean)}
    return (
        result.multiplicity == nested and result.boolean_incidence == expected_boolean
    )
