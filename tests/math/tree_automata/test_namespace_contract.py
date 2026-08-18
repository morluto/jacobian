"""Owner-local exact public API contract for tree_automata."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.tree_automata")
    expected = (
        "BottomUpTreeAutomaton",
        "RankedTree",
        "TreeAutomatonTransition",
        "accepted_tree_count",
        "run_tree_automaton",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
