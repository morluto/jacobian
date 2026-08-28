"""Combinatorial-map operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
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
    _op(
        "combinatorial_map.faces.compute",
        "Compute the face-orbit family of a combinatorial map",
        "Return the complete face-orbit family, the per-dart face assignment, "
        "and the dart-successor permutation along each face. Every dart occurs "
        "in exactly one facial walk.",
        FacesRequest,
        FacesResult,
        compute_faces,
        "combinatorial-map",
        "faces",
        "exact",
        examples=(
            example(
                "four_cycle_faces",
                "Faces of a 4-cycle embedded on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    _op(
        "combinatorial_map.euler_characteristic.compute",
        "Compute per-component and total Euler characteristic",
        "Return V, E, F and chi = V - E + F for each connected component and "
        "in total under the disconnected-surface convention (each component is "
        "an independent closed surface).",
        EulerCharacteristicRequest,
        EulerCharacteristicResult,
        compute_euler_characteristic,
        "combinatorial-map",
        "euler-characteristic",
        "exact",
        examples=(
            example(
                "four_cycle_euler",
                "Euler characteristic of a 4-cycle on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    _op(
        "combinatorial_map.orientable_genus.compute",
        "Compute per-component and total orientable genus",
        "For each connected component, compute g = (2 - chi) / 2 under the "
        "orientable cellular-map convention, plus the total genus of the "
        "disjoint union. The result is an exact nonnegative integer for a "
        "valid orientable combinatorial map.",
        OrientableGenusRequest,
        OrientableGenusResult,
        compute_orientable_genus,
        "combinatorial-map",
        "genus",
        "exact",
        examples=(
            example(
                "four_cycle_genus",
                "Genus of a 4-cycle on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    _op(
        "combinatorial_map.orientation_reverse.compute",
        "Reverse every local cyclic order",
        "Reverse every local cyclic order and return the resulting combinatorial "
        "map together with the induced bijection on faces. Applying the "
        "operation twice returns the original map exactly under canonical "
        "transport convention.",
        OrientationReverseRequest,
        OrientationReverseResult,
        compute_orientation_reverse,
        "combinatorial-map",
        "orientation",
        "exact",
        examples=(
            example(
                "four_cycle_orientation_reverse",
                "Reverse the orientation of a 4-cycle on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    _op(
        "combinatorial_map.connected_components.compute",
        "Return the component partition of vertices, darts, and faces",
        "Return the vertex, dart, and face component partition of the supplied "
        "combinatorial map. Useful when downstream topology treats components "
        "independently.",
        ConnectedComponentsRequest,
        ConnectedComponentsResult,
        compute_connected_components,
        "combinatorial-map",
        "components",
        "exact",
        examples=(
            example(
                "four_cycle_components",
                "Component partition of a 4-cycle on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    _op(
        "combinatorial_map.dual.compute",
        "Compute the exact embedded dual",
        "Return the exact embedded dual: one dual vertex per primal face, one "
        "dual dart per primal dart, dual reversal inherited from primal "
        "reversal, and dual tail/head determined by the two incident face "
        "sides. The dual of a bridge becomes a loop; parallel dual edges are "
        "retained with identity.",
        DualRequest,
        DualResult,
        compute_dual,
        "combinatorial-map",
        "dual",
        "exact",
        examples=(
            example(
                "four_cycle_dual",
                "Dual of a 4-cycle on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
    _op(
        "combinatorial_map.vertex_face_incidence.compute",
        "Compute the vertex-face incidence structure",
        "Return the exact finite incidence structure between primal vertices "
        "and faces, including multiplicity when one vertex occurs several "
        "times on a facial boundary, plus the boolean per-vertex face set.",
        VertexFaceIncidenceRequest,
        VertexFaceIncidenceResult,
        compute_vertex_face_incidence,
        "combinatorial-map",
        "incidence",
        "exact",
        examples=(
            example(
                "four_cycle_incidence",
                "Vertex-face incidence of a 4-cycle on the sphere.",
                {"map": _CYCLE_MAP["map"]},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
