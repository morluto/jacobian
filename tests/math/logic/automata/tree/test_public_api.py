"""Exact public API contract for jacobian.math.logic.automata.tree."""

from __future__ import annotations

from jacobian.math.logic.automata import tree as tree_automata


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the tree_automata public API."""
    expected = (
        "AcceptedTreeHeightProfileRequest",
        "AcceptedTreeHeightProfileResult",
        "BottomUpTreeAutomaton",
        "CompleteDeterministicBottomUpTreeAutomaton",
        "DeterministicBottomUpTreeAutomaton",
        "NondeterministicRunCountsRequest",
        "NondeterministicRunCountsResult",
        "RankedTree",
        "RankedTreePositionsRequest",
        "RankedTreePositionsResult",
        "RankedTreeSubtreeRequest",
        "RankedTreeSubtreeResult",
        "ReachableStateProfile",
        "RegularTreeGrammar",
        "RegularTreeProduction",
        "TreeAutomatonBooleanProductRequest",
        "TreeAutomatonBooleanProductResult",
        "TreeAutomatonComplementResult",
        "TreeAutomatonCompletionRequest",
        "TreeAutomatonCompletionResult",
        "TreeAutomatonMinimizeRequest",
        "TreeAutomatonMinimizeResult",
        "TreeAutomatonTransition",
        "accepted_tree_count",
        "accepted_tree_height_profile",
        "boolean_product_tree_automata",
        "complement_tree_automaton",
        "complete_deterministic_tree_automaton",
        "determinize_tree_automaton",
        "minimize_tree_automaton",
        "nondeterministic_run_counts",
        "ranked_tree_positions",
        "ranked_tree_subtree",
        "reachable_state_profile",
        "regular_tree_grammar_to_automaton",
        "run_tree_automaton",
        "trim_tree_automaton",
        "verify_accepted_tree_count",
        "verify_determinization",
        "verify_reachable_state_profile",
        "verify_tree_run",
        "verify_trim_tree_automaton",
    )
    assert tuple(tree_automata.__all__) == expected
    assert len(tree_automata.__all__) == len(set(tree_automata.__all__))
    assert all(not name.startswith("_") for name in tree_automata.__all__)
    assert all(hasattr(tree_automata, name) for name in tree_automata.__all__)
