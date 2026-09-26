"""Exact public API contract for jacobian.math.logic.automata.tree."""

from __future__ import annotations

from jacobian.math.logic.automata import tree as tree_automata


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the tree_automata public API."""
    expected = (
        "BottomUpTreeAutomaton",
        "CompleteDeterministicBottomUpTreeAutomaton",
        "DeterministicBottomUpTreeAutomaton",
        "FiniteTreeContext",
        "RankedTree",
        "RankedTreePositionsRequest",
        "RankedTreePositionsResult",
        "RankedTreeSubtreeRequest",
        "RankedTreeSubtreeResult",
        "ReachableStateProfile",
        "TreeAutomatonBooleanProductRequest",
        "TreeAutomatonBooleanProductResult",
        "TreeAutomatonComplementResult",
        "TreeAutomatonCompletionRequest",
        "TreeAutomatonCompletionResult",
        "TreeAutomatonMinimizeRequest",
        "TreeAutomatonMinimizeResult",
        "TreeAutomatonTransition",
        "TreeContextFrame",
        "TreeContextPlugRequest",
        "TreeContextPlugResult",
        "TreeContextStateMapRequest",
        "TreeContextStateMapResult",
        "TreeContextTransformation",
        "TreeContextTransformationMonoidRequest",
        "TreeContextTransformationMonoidResult",
        "accepted_tree_count",
        "boolean_product_tree_automata",
        "complement_tree_automaton",
        "complete_deterministic_tree_automaton",
        "determinize_tree_automaton",
        "map_tree_context_states",
        "minimize_tree_automaton",
        "plug_tree_context_operation",
        "ranked_tree_positions",
        "ranked_tree_subtree",
        "reachable_state_profile",
        "run_tree_automaton",
        "tree_context_transformation_monoid",
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
