"""Supported native combinatorial-map API."""

from jacobian.math.topology.combinatorial_maps.operations import (
    connected_components,
    connected_components_vertices,
    dual_map,
    euler_characteristic,
    face_orbits,
    orientable_genus,
    orientation_reverse,
    rotation_successor,
    vertex_face_incidence,
)
from jacobian.math.topology.combinatorial_maps.values import (
    FacialWalk,
    FiniteCombinatorialMap,
)

__all__ = [
    "FacialWalk",
    "FiniteCombinatorialMap",
    "connected_components",
    "connected_components_vertices",
    "dual_map",
    "euler_characteristic",
    "face_orbits",
    "orientable_genus",
    "orientation_reverse",
    "rotation_successor",
    "vertex_face_incidence",
]
