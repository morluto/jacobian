"""Tests for bottom-up tree-automaton trim (#1922)."""

from __future__ import annotations

from jacobian.math.logic.automata.tree._models import (
    TreeAutomatonTrimRequest,
    TreeAutomatonTrimResult,
)
from jacobian.math.logic.automata.tree._tools import compute_tree_automaton_trim
from jacobian.math.logic.automata.tree.operations import (
    run_tree_automaton,
    trim_tree_automaton,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    RankedTree,
)


def _automaton(**overrides: object) -> BottomUpTreeAutomaton:
    payload = {
        "state_count": 2,
        "arity": [0, 2],
        "transitions": [
            {"symbol": 0, "child_states": [], "target_state": 0},
            {"symbol": 1, "child_states": [0, 0], "target_state": 0},
            {"symbol": 1, "child_states": [0, 1], "target_state": 1},
            {"symbol": 1, "child_states": [1, 0], "target_state": 1},
            {"symbol": 1, "child_states": [1, 1], "target_state": 1},
        ],
        "final_states": [0],
    }
    payload.update(overrides)
    return BottomUpTreeAutomaton.model_validate(payload)


LEAF = RankedTree.model_validate({"symbol": 0, "children": []})
FORK = RankedTree.model_validate(
    {
        "symbol": 1,
        "children": [
            {"symbol": 0, "children": []},
            {"symbol": 0, "children": []},
        ],
    }
)


class TestKnownAnswer:
    def test_drops_unreachable_state(self) -> None:
        result = trim_tree_automaton(_automaton())
        assert isinstance(result, TreeAutomatonTrimResult)
        assert result.kept_states == (0,)
        assert result.dropped_states == (1,)
        assert result.old_to_new == (0, -1)
        assert result.new_to_old == (0,)
        assert not result.empty_language
        assert result.trimmed.state_count == 1
        assert result.trimmed.final_states == (0,)
        assert len(result.witnesses) == 1

    def test_identity_trim(self) -> None:
        automaton = _automaton(
            transitions=[
                {"symbol": 0, "child_states": [], "target_state": 0},
                {"symbol": 0, "child_states": [], "target_state": 1},
                {"symbol": 1, "child_states": [0, 1], "target_state": 0},
            ],
            final_states=[0],
        )
        result = trim_tree_automaton(automaton)
        assert result.kept_states == (0, 1)
        assert result.dropped_states == ()
        assert result.old_to_new == (0, 1)
        assert result.trimmed.state_count == 2


class TestBoundaryDegenerate:
    def test_empty_language_canonical_trim(self) -> None:
        # No nullary transition reaches the final state: empty language.
        automaton = _automaton(
            transitions=[
                {"symbol": 1, "child_states": [0, 0], "target_state": 0},
            ],
            final_states=[0],
        )
        result = trim_tree_automaton(automaton)
        assert result.empty_language
        assert result.kept_states == ()
        assert result.trimmed.state_count == 1
        assert result.trimmed.transitions == ()
        assert result.trimmed.final_states == ()
        assert result.witnesses == ()

    def test_unreachable_final_state(self) -> None:
        automaton = _automaton(
            transitions=[
                {"symbol": 0, "child_states": [], "target_state": 0},
            ],
            final_states=[1],
        )
        result = trim_tree_automaton(automaton)
        assert result.empty_language
        assert result.kept_states == ()


class TestAdversarial:
    def test_unproductive_reachable_state_dropped(self) -> None:
        # State 1 is reachable but never occurs inside an accepting run.
        automaton = _automaton(
            transitions=[
                {"symbol": 0, "child_states": [], "target_state": 0},
                {"symbol": 0, "child_states": [], "target_state": 1},
                {"symbol": 1, "child_states": [0, 0], "target_state": 0},
            ],
            final_states=[0],
        )
        result = trim_tree_automaton(automaton)
        assert result.kept_states == (0,)
        assert result.dropped_states == (1,)


class TestDefiningInvariant:
    def test_trim_preserves_acceptance(self) -> None:
        automaton = _automaton()
        result = trim_tree_automaton(automaton)
        for tree in (LEAF, FORK):
            source_states = run_tree_automaton(automaton, tree)
            trimmed_states = run_tree_automaton(result.trimmed, tree)
            source_accepted = bool(source_states & set(automaton.final_states))
            trimmed_accepted = bool(trimmed_states & set(result.trimmed.final_states))
            assert source_accepted == trimmed_accepted
            assert (
                run_tree_automaton(result.trimmed, tree)
                == {
                    result.old_to_new[state]
                    for state in source_states
                    if result.old_to_new[state] >= 0
                }
                or not source_accepted
            )

    def test_witnesses_replay_on_trimmed(self) -> None:
        automaton = _automaton()
        result = trim_tree_automaton(automaton)
        assert tuple(w.state for w in result.witnesses) == tuple(
            range(result.trimmed.state_count)
        )
        for witness in result.witnesses:
            assert witness.state in run_tree_automaton(result.trimmed, witness.tree)

    def test_old_new_maps_are_inverse(self) -> None:
        result = trim_tree_automaton(_automaton())
        assert result.new_to_old == result.kept_states
        for old, new in enumerate(result.old_to_new):
            assert (new == -1) == (old in result.dropped_states)
            if new >= 0:
                assert result.new_to_old[new] == old


class TestNativeCatalogParity:
    def test_native_matches_catalog(self) -> None:
        request = TreeAutomatonTrimRequest(automaton=_automaton())
        assert compute_tree_automaton_trim(request) == trim_tree_automaton(
            request.automaton
        )

    def test_empty_language_parity(self) -> None:
        request = TreeAutomatonTrimRequest(
            automaton=_automaton(
                transitions=[
                    {"symbol": 1, "child_states": [0, 0], "target_state": 0},
                ],
                final_states=[0],
            )
        )
        assert compute_tree_automaton_trim(request) == trim_tree_automaton(
            request.automaton
        )
