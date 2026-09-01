"""Combinatorial-map operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
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


# A 4-cycle on the sphere: 4 vertices, 4 edges, 2 faces.
# Vertices 0-3 in a ring. Edge i connects vertex i to vertex (i+1) mod 4.
_CYCLE_DARTS = [
    [0, 1, 1],
    [1, 0, 0],
    [1, 2, 3],
    [2, 1, 2],
    [2, 3, 5],
    [3, 2, 4],
    [3, 0, 7],
    [0, 3, 6],
]
_CYCLE_ROTATIONS = [
    [0, 7],
    [1, 2],
    [3, 4],
    [5, 6],
]
_CYCLE_MAP = {
    "map": {
        "vertex_count": 4,
        "darts": _CYCLE_DARTS,
        "rotations": _CYCLE_ROTATIONS,
    }
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="combinatorial_map.faces.compute",
        title="Compute the face-orbit family of a combinatorial map",
        description="Return the complete face-orbit family, the per-dart face assignment, "
        "and the dart-successor permutation along each face. Every dart occurs "
        "in exactly one facial walk.",
        request_type=FacesRequest,
        result_type=FacesResult,
        run=compute_faces,
        tags=("combinatorial-map", "faces", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_faces",
                description="Faces of a 4-cycle embedded on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorial_map.euler_characteristic.compute",
        title="Compute per-component and total Euler characteristic",
        description="Return V, E, F and chi = V - E + F for each connected component and "
        "in total under the disconnected-surface convention (each component is "
        "an independent closed surface).",
        request_type=EulerCharacteristicRequest,
        result_type=EulerCharacteristicResult,
        run=compute_euler_characteristic,
        tags=("combinatorial-map", "euler-characteristic", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_euler",
                description="Euler characteristic of a 4-cycle on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorial_map.orientable_genus.compute",
        title="Compute per-component and total orientable genus",
        description="For each connected component, compute g = (2 - chi) / 2 under the "
        "orientable cellular-map convention, plus the total genus of the "
        "disjoint union. The result is an exact nonnegative integer for a "
        "valid orientable combinatorial map.",
        request_type=OrientableGenusRequest,
        result_type=OrientableGenusResult,
        run=compute_orientable_genus,
        tags=("combinatorial-map", "genus", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_genus",
                description="Genus of a 4-cycle on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorial_map.orientation_reverse.compute",
        title="Reverse every local cyclic order",
        description="Reverse every local cyclic order and return the resulting combinatorial "
        "map together with the induced bijection on faces. Applying the "
        "operation twice returns the original map exactly under canonical "
        "transport convention.",
        request_type=OrientationReverseRequest,
        result_type=OrientationReverseResult,
        run=compute_orientation_reverse,
        tags=("combinatorial-map", "orientation", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_orientation_reverse",
                description="Reverse the orientation of a 4-cycle on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorial_map.connected_components.compute",
        title="Return the component partition of vertices, darts, and faces",
        description="Return the vertex, dart, and face component partition of the supplied "
        "combinatorial map. Useful when downstream topology treats components "
        "independently.",
        request_type=ConnectedComponentsRequest,
        result_type=ConnectedComponentsResult,
        run=compute_connected_components,
        tags=("combinatorial-map", "components", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_components",
                description="Component partition of a 4-cycle on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorial_map.dual.compute",
        title="Compute the exact embedded dual",
        description="Return the exact embedded dual: one dual vertex per primal face, one "
        "dual dart per primal dart, dual reversal inherited from primal "
        "reversal, and dual tail/head determined by the two incident face "
        "sides. The dual of a bridge becomes a loop; parallel dual edges are "
        "retained with identity.",
        request_type=DualRequest,
        result_type=DualResult,
        run=compute_dual,
        tags=("combinatorial-map", "dual", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_dual",
                description="Dual of a 4-cycle on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    MathTool(
        operation_id="combinatorial_map.vertex_face_incidence.compute",
        title="Compute the vertex-face incidence structure",
        description="Return the exact finite incidence structure between primal vertices "
        "and faces, including multiplicity when one vertex occurs several "
        "times on a facial boundary, plus the boolean per-vertex face set.",
        request_type=VertexFaceIncidenceRequest,
        result_type=VertexFaceIncidenceResult,
        run=compute_vertex_face_incidence,
        tags=("combinatorial-map", "incidence", "exact"),
        examples=(
            OperationExample(
                name="four_cycle_incidence",
                description="Vertex-face incidence of a 4-cycle on the sphere.",
                input={"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
