"""Exact public API contract for jacobian.math.finite_topology."""

from __future__ import annotations

from jacobian.math import finite_topology


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the finite_topology public API."""
    expected = (
        "BeatPointAnalysis",
        "BeatPointWitness",
        "ContinuityAnalysis",
        "FiniteTopology",
        "PointMap",
        "beat_points",
        "closure",
        "connected_components",
        "continuity",
        "interior",
        "is_continuous",
        "is_t0",
        "minimal_open_neighborhoods",
        "specialization_preorder",
    )
    assert tuple(finite_topology.__all__) == expected
    assert len(finite_topology.__all__) == len(set(finite_topology.__all__))
    assert all(not name.startswith("_") for name in finite_topology.__all__)
    assert all(hasattr(finite_topology, name) for name in finite_topology.__all__)
