"""Owner-local exact public API contract for symbolic_dynamics."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.symbolic_dynamics")
    expected = (
        "AdjacencyShift",
        "BlockPresentation",
        "ForbiddenBlockShift",
        "LabeledTransition",
        "adjacency_shift",
        "block_language",
        "finite_type_presentation",
        "higher_block_presentation",
        "normalize_forbidden_blocks",
        "periodic_point_profile",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
