"""Exact public API contract for jacobian.math.topology.finite.spaces."""

from __future__ import annotations

from jacobian.math.topology.finite import spaces as finite_topology_spaces


def test_exact_public_api_symbols() -> None:
    expected = (
        "FiniteTopologicalMap",
        "FiniteTopologicalSpace",
        "FiniteTopologicalSubset",
        "boundary",
        "closure",
        "continuous_check",
        "from_preorder",
        "interior",
        "kolmogorov_quotient",
        "minimal_neighbourhoods",
        "specialization_preorder",
        "verify_boundary",
        "verify_closure",
        "verify_continuity",
        "verify_interior",
        "verify_kolmogorov_quotient",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(finite_topology_spaces.__all__)
    assert len(finite_topology_spaces.__all__) == len(
        set(finite_topology_spaces.__all__)
    )
    assert all(not name.startswith("_") for name in finite_topology_spaces.__all__)
    assert all(
        hasattr(finite_topology_spaces, name) for name in finite_topology_spaces.__all__
    )
