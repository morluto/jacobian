"""Owner-local exact public API contract for finite_topology."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.finite_topology")
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
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
