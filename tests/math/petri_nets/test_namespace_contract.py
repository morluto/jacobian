"""Owner-local exact public API contract for petri_nets."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.petri_nets")
    expected = (
        "Marking",
        "PetriNet",
        "compute_incidence_matrix",
        "enabled_transitions",
        "fire_transition",
        "reachability_graph",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
