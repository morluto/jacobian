"""Public native kernels re-exported by the supported native API."""

from jacobian.math.combinatorial_maps.operations_module import (
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

__all__ = [
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
