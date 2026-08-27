"""Focused accounting evidence for tree-automaton reachability."""

from __future__ import annotations

from unittest.mock import patch

from tests.fixtures.accounting import assert_charged_work_parity

from jacobian.math.tree_automata import values
from jacobian.math.tree_automata._models import TreeAutomatonReachabilityRequest
from jacobian.math.tree_automata._operations import compute_tree_automaton_reachability
from jacobian.math.tree_automata.values import (
    BottomUpTreeAutomaton,
    TreeAutomatonTransition,
)


def _automaton() -> BottomUpTreeAutomaton:
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


def test_public_reachability_path_charges_each_saturation_pass() -> None:
    with patch.object(
        values, "_priced_saturation", wraps=values._priced_saturation
    ) as run:
        result = compute_tree_automaton_reachability(
            TreeAutomatonReachabilityRequest(automaton=_automaton())
        )

    assert result.reachable_states == (0,)
    assert_charged_work_parity(
        charged={"saturation": 3}, executed={"saturation": run.call_count}
    )


def test_schema_describes_the_three_priced_passes() -> None:
    schema = TreeAutomatonReachabilityRequest.model_json_schema()
    description = schema["properties"]["automaton"]["description"]

    assert "three passes" in description
    assert "four passes" not in description
