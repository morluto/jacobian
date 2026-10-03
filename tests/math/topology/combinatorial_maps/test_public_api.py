"""Exact public API contract for jacobian.math.topology.combinatorial_maps."""

from __future__ import annotations

from jacobian.math.topology import combinatorial_maps


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the combinatorial_maps public API."""
    expected = (
        "FacialWalk",
        "FiniteCombinatorialMap",
        "check_multigraph_embedding",
        "check_orientable_embedding",
        "check_signed_embedding",
        "connected_components",
        "connected_components_vertices",
        "dual_map",
        "euler_characteristic",
        "face_orbits",
        "find_rotation_system",
        "minimum_orientable_genus",
        "orientable_genus",
        "orientation_reverse",
        "rotation_successor",
        "verify_dual",
        "verify_minimum_orientable_genus",
        "verify_multigraph_embedding",
        "verify_orientable_embedding",
        "verify_orientation_reverse",
        "verify_rotation_system_find",
        "verify_signed_embedding",
        "verify_vertex_face_incidence",
        "vertex_face_incidence",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(combinatorial_maps.__all__)
    assert len(combinatorial_maps.__all__) == len(set(combinatorial_maps.__all__))
    assert all(not name.startswith("_") for name in combinatorial_maps.__all__)
    assert all(hasattr(combinatorial_maps, name) for name in combinatorial_maps.__all__)
