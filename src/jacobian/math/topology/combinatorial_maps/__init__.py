"""Supported native combinatorial-map API."""

from jacobian.math.topology.combinatorial_maps.operations import (
    check_orientable_embedding,
    check_signed_embedding,
    connected_components,
    connected_components_vertices,
    dual_map,
    euler_characteristic,
    face_orbits,
    orientable_genus,
    orientation_reverse,
    rotation_successor,
    verify_dual,
    verify_orientable_embedding,
    verify_orientation_reverse,
    verify_signed_embedding,
    verify_vertex_face_incidence,
    vertex_face_incidence,
)
from jacobian.math.topology.combinatorial_maps.values import (
    FacialWalk,
    FiniteCombinatorialMap,
)

__all__ = [
    "FacialWalk",
    "FiniteCombinatorialMap",
    "check_orientable_embedding",
    "check_signed_embedding",
    "connected_components",
    "connected_components_vertices",
    "dual_map",
    "euler_characteristic",
    "face_orbits",
    "orientable_genus",
    "orientation_reverse",
    "rotation_successor",
    "verify_dual",
    "verify_orientable_embedding",
    "verify_orientation_reverse",
    "verify_signed_embedding",
    "verify_vertex_face_incidence",
    "vertex_face_incidence",
]
