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
    OrientableEmbeddingCheckRequest,
    OrientableEmbeddingCheckResult,
    OrientableGenusRequest,
    OrientableGenusResult,
    OrientationReverseRequest,
    OrientationReverseResult,
    RotationSystemFindRequest,
    RotationSystemFindResult,
    VertexFaceIncidenceRequest,
    VertexFaceIncidenceResult,
)
from jacobian.math.topology.combinatorial_maps.operations import (
    check_orientable_embedding,
    connected_components,
    dual_map,
    euler_characteristic,
    face_orbits,
    find_rotation_system,
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


def compute_orientable_embedding_check(
    request: OrientableEmbeddingCheckRequest,
) -> OrientableEmbeddingCheckResult:
    return check_orientable_embedding(request.graph, request.rotations)


def compute_rotation_system_find(
    request: RotationSystemFindRequest,
) -> RotationSystemFindResult:
    return find_rotation_system(
        request.graph, request.max_genus, request.max_candidates
    )


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
    MathTool(
        operation_id="graph.embedding.orientable.check",
        title="Check an orientable cellular graph embedding",
        description=(
            "Check one supplied rotation system of a connected bounded simple "
            "graph as a cellular embedding in a closed orientable surface. The "
            "result carries canonical local rotations, the dart permutations "
            "alpha, sigma, and phi = alpha . sigma, complete face walks, "
            "chi = V - E + F, and the exact orientable genus g = (2 - chi)/2. "
            "Unsigned rotation systems are orientable by convention; an "
            "INVALID_EMBEDDING carries the first rotation obstruction. This is a "
            "checker for a supplied rotation system, not a genus minimizer."
        ),
        request_type=OrientableEmbeddingCheckRequest,
        result_type=OrientableEmbeddingCheckResult,
        run=compute_orientable_embedding_check,
        tags=("graph", "embedding", "genus", "orientable", "exact"),
        discovery_terms=(
            "rotation system",
            "orientable embedding",
            "graph genus",
            "combinatorial map",
        ),
        examples=(
            OperationExample(
                name="tetrahedron_sphere",
                description=(
                    "A planar rotation system of K4; rotations list incident "
                    "edge indices at each vertex in edge-list order."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["a", "d"],
                            ["b", "c"],
                            ["b", "d"],
                            ["c", "d"],
                        ],
                    },
                    "rotations": [[0, 1, 2], [0, 4, 3], [1, 3, 5], [2, 5, 4]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.embedding.rotation_system.find",
        title="Find a rotation system of bounded orientable genus",
        description=(
            "Enumerate rotation systems of a connected bounded simple graph "
            "in deterministic vertex order (first rotation entry fixed, so "
            "each cellular embedding appears once up to cyclic shifts) and "
            "decide each one with the exact orientable checker. FOUND carries "
            "the first system of genus at most max_genus with its checker "
            "certificate; EXHAUSTED carries the receipt that every one of the "
            "exact total_candidates systems was examined; UNKNOWN carries a "
            "bounded reason when the candidate budget runs out or the graph "
            "is disconnected. A truncated search never yields a negative "
            "conclusion."
        ),
        request_type=RotationSystemFindRequest,
        result_type=RotationSystemFindResult,
        run=compute_rotation_system_find,
        tags=("graph", "embedding", "genus", "search", "exact"),
        discovery_terms=(
            "rotation system",
            "graph genus",
            "low genus embedding",
            "bounded search",
        ),
        examples=(
            OperationExample(
                name="k4_planar_rotation",
                description=(
                    "K4 is planar: the bounded search finds a genus-zero "
                    "rotation system."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "c"],
                            ["a", "d"],
                            ["b", "c"],
                            ["b", "d"],
                            ["c", "d"],
                        ],
                    },
                    "max_genus": 0,
                    "max_candidates": 1000,
                },
            ),
            OperationExample(
                name="k33_nonplanar_exhaustion",
                description=(
                    "K3,3 is nonplanar: all 64 rotation systems are examined "
                    "and none has genus zero."
                ),
                input={
                    "graph": {
                        "vertices": ["a", "b", "c", "x", "y", "z"],
                        "edges": [
                            ["a", "x"],
                            ["a", "y"],
                            ["a", "z"],
                            ["b", "x"],
                            ["b", "y"],
                            ["b", "z"],
                            ["c", "x"],
                            ["c", "y"],
                            ["c", "z"],
                        ],
                    },
                    "max_genus": 0,
                    "max_candidates": 1000,
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
