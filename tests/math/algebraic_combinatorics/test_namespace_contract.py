"""Owner-local exact public API contract for algebraic_combinatorics."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.algebraic_combinatorics")
    expected = (
        "conjugate_partition",
        "hook_lengths",
        "standard_young_tableaux_count",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
