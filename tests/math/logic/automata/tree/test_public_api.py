"""Exact public API contract for jacobian.math.logic.automata.tree."""

from __future__ import annotations

from jacobian.math.logic.automata import tree as tree_automata


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the tree_automata public API."""
    expected = (
        "BottomUpTreeAutomaton",
        "RankedTree",
        "ReachableStateProfile",
        "TreeAutomatonTransition",
        "accepted_tree_count",
        "reachable_state_profile",
        "run_tree_automaton",
    )
    assert tuple(tree_automata.__all__) == expected
    assert len(tree_automata.__all__) == len(set(tree_automata.__all__))
    assert all(not name.startswith("_") for name in tree_automata.__all__)
    assert all(hasattr(tree_automata, name) for name in tree_automata.__all__)
