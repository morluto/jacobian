"""Exact public API contract for jacobian.math.finite_topology_spaces."""

from __future__ import annotations

from jacobian.math import finite_topology_spaces


def test_exact_public_api_symbols() -> None:
    expected = (
        "FiniteTopologicalMap",
        "FiniteTopologicalSpace",
        "boundary",
        "closure",
        "continuous_check",
        "from_preorder",
        "interior",
        "kolmogorov_quotient",
        "minimal_neighbourhoods",
        "specialization_preorder",
    )
    assert tuple(finite_topology_spaces.__all__) == expected
    assert len(finite_topology_spaces.__all__) == len(
        set(finite_topology_spaces.__all__)
    )
    assert all(not name.startswith("_") for name in finite_topology_spaces.__all__)
    assert all(
        hasattr(finite_topology_spaces, name) for name in finite_topology_spaces.__all__
    )
