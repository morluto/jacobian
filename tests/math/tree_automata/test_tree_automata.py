"""Tests for tree automaton operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.tree_automata._models import (
    AcceptedTreeCountRequest,
    TreeRunRequest,
)
from jacobian.math.tree_automata._operations import (
    compute_accepted_tree_count,
    compute_tree_run,
)
from jacobian.math.tree_automata.operations import (
    run_tree_automaton,
)
from jacobian.math.tree_automata.values import (
    BottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


# Helpers
def _leaf() -> RankedTree:
    return RankedTree(symbol=0, children=())


def _node(left: RankedTree, right: RankedTree) -> RankedTree:
    return RankedTree(symbol=1, children=(left, right))


def _simple_automaton() -> BottomUpTreeAutomaton:
    """2-state automaton: accepts balanced binary trees."""
    return BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(1, 0), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(0, 1), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(1, 1), target_state=1),
        ),
        final_states=(0,),
    )


class TestRun:
    def test_accepted_leaf(self):
        automaton = _simple_automaton()
        tree = _leaf()
        states = run_tree_automaton(automaton, tree)
        assert states == {0}

    def test_accepted_balanced_tree(self):
        automaton = _simple_automaton()
        tree = _node(_leaf(), _leaf())
        states = run_tree_automaton(automaton, tree)
        assert states == {0}

    def test_balanced_tree_state1(self):
        # With the simple automaton, f(a, a) -> state 0 (balanced)
        automaton = _simple_automaton()
        tree = _node(_leaf(), _leaf())
        states = run_tree_automaton(automaton, tree)
        assert states == {0}

    def test_run_request_accepts(self):
        automaton = _simple_automaton()
        tree = _node(_leaf(), _leaf())
        result = compute_tree_run(TreeRunRequest(automaton=automaton, tree=tree))
        assert result.accepted is True
        assert result.root_states == (0,)

    def test_run_request_rejects(self):
        # Use automaton where state 1 is final but not state 0
        automaton = BottomUpTreeAutomaton(
            state_count=2,
            arity=(0, 2),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
                TreeAutomatonTransition(symbol=1, child_states=(1, 1), target_state=0),
            ),
            final_states=(0,),
        )
        # Tree f(a, a) with a -> state 1, f(1, 1) -> state 0 (accepted)
        tree = _node(_leaf(), _leaf())
        result = compute_tree_run(TreeRunRequest(automaton=automaton, tree=tree))
        assert result.accepted is True
        assert result.root_states == (0,)
        # Leaf alone: a -> state 1 (not final, rejected)
        leaf_result = compute_tree_run(
            TreeRunRequest(automaton=automaton, tree=_leaf())
        )
        assert leaf_result.accepted is False
        assert leaf_result.root_states == (1,)


class TestAcceptedTreeCount:
    def test_count_size_1(self):
        automaton = _simple_automaton()
        result = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=1)
        )
        assert result.count == 1

    def test_count_size_3(self):
        automaton = _simple_automaton()
        result = compute_accepted_tree_count(
            AcceptedTreeCountRequest(automaton=automaton, tree_size=3)
        )
        assert result.count == 1


class TestValidation:
    def test_arity_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0, 2),
                transitions=(
                    TreeAutomatonTransition(
                        symbol=0, child_states=(0,), target_state=0
                    ),
                ),
                final_states=(0,),
            )

    def test_symbol_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0,),
                transitions=(
                    TreeAutomatonTransition(
                        symbol=5, child_states=(), target_state=0
                    ),
                ),
                final_states=(0,),
            )

    def test_final_state_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            BottomUpTreeAutomaton(
                state_count=1,
                arity=(0,),
                transitions=(),
                final_states=(5,),
            )
