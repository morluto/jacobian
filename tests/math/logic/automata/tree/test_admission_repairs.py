"""Admission and decoding regressions for the tree-automata owner.

Each test fails against the corresponding defect and passes on the repair; the
negative control is this file run against unmodified `main`.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.tree import (
    accepted_tree_height_profile,
    tree_automaton_to_regular_tree_grammar,
    tree_language_profile,
)
from jacobian.math.logic.automata.tree import operations as tree_operations
from jacobian.math.logic.automata.tree._models import TreeLanguageProfile
from jacobian.math.logic.automata.tree.contexts import (
    FiniteTreeContext,
    TreeContextFrame,
)
from jacobian.math.logic.automata.tree.values import (
    MAX_REACHABILITY_WITNESS_NODES,
    MAX_RUN_TREE_DEPTH,
    MAX_TA_ARITY,
    MAX_TA_STATES,
    MAX_TA_TRANSITIONS,
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


def _complete_binary_rows(target: int) -> tuple[TreeAutomatonTransition, ...]:
    """Every symbol and child-state tuple of the two-state binary automaton."""
    return tuple(
        TreeAutomatonTransition(symbol=symbol, child_states=combo, target_state=target)
        for symbol, arity in enumerate((0, 2))
        for combo in product((0, 1), repeat=arity)
    )


def test_empty_language_height_profile_is_not_refused_by_the_digit_bound() -> None:
    """An empty language has a zero profile; the bound must not refuse it.

    Two states, arity (0, 2), and no rule reaching state 0, so no tree is ever
    accepted and the exact answer is (max_height + 1) zeros. The all-trees digit
    bound is decided before the recurrence, so it refused this request.
    """
    empty = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 2),
        transitions=_complete_binary_rows(target=1),
        final_states=(0,),
    )

    assert accepted_tree_height_profile(empty, 20) == (0,) * 21

    # A language with a reachable final state still grows and is still bounded.
    reachable = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            *(
                TreeAutomatonTransition(symbol=1, child_states=combo, target_state=1)
                for combo in product((0, 1), repeat=2)
            ),
        ),
        final_states=(0,),
    )
    # Only nullary trees reach a final state here, so every height counts one.
    assert accepted_tree_height_profile(reachable, 3) == (1, 1, 1, 1)


def test_height_profile_still_admits_its_recurrence_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 2),
        transitions=_complete_binary_rows(target=1),
        final_states=(0,),
    )
    monkeypatch.setattr(tree_operations, "MAX_TREE_AUTOMATON_WORK", 1, raising=False)
    with pytest.raises(OperationResourceAdmissionError) as error:
        accepted_tree_height_profile(empty, 5)
    assert error.value.errors()[0]["type"] == "tree_automata.height_count_work_bound"

    assert error.value.errors()[0]["type"] == "tree_automata.height_count_work_bound"


def test_productivity_saturation_is_admitted_before_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saturation cost depends on transition order, so it must be charged.

    The loop rescans every row once per newly productive state, so its pass
    count is bounded by the state count rather than by the table's size. The
    sibling productivity preflight already prices that shape; the grammar
    conversion charged only ``O(rows)``.
    """
    automaton = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
        ),
        final_states=(0,),
    )
    monkeypatch.setattr(
        tree_operations, "MAX_TREE_GRAMMAR_CONVERSION_WORK", 1, raising=False
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        tree_automaton_to_regular_tree_grammar(automaton)
    assert error.value.errors()[0]["type"] == "tree_automata.grammar_conversion_bound"

    monkeypatch.undo()
    calls: list[int] = []
    admitted = tree_operations.__dict__["_admit_productivity_saturation"]

    def recording(value: BottomUpTreeAutomaton) -> None:
        calls.append(1)
        admitted(value)

    monkeypatch.setattr(tree_operations, "_admit_productivity_saturation", recording)
    grammar = tree_automaton_to_regular_tree_grammar(
        _chain_automaton(states=8, rows=64)
    )
    assert calls == [1]
    assert grammar.productions


def _chain_automaton(states: int, rows: int) -> BottomUpTreeAutomaton:
    """A reverse dependency chain, so each new productive state costs a pass."""
    arity = (0, 1, 3)
    transitions: list[TreeAutomatonTransition] = [
        TreeAutomatonTransition(symbol=1, child_states=(index + 1,), target_state=index)
        for index in range(states - 1)
    ]
    seen = {(row.symbol, row.child_states) for row in transitions}
    for child in range(states):
        for width in (3,):
            key = (2, (child,) * width)
            if key in seen or len(transitions) == rows:
                continue
            seen.add(key)
            transitions.append(
                TreeAutomatonTransition(
                    symbol=key[0], child_states=key[1], target_state=0
                )
            )
    transitions.append(
        TreeAutomatonTransition(symbol=0, child_states=(), target_state=states - 1)
    )
    return BottomUpTreeAutomaton(
        state_count=states,
        arity=arity,
        transitions=tuple(transitions),
        final_states=(states - 1,),
    )


def test_productivity_saturation_crosses_a_cancellation_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long saturation must observe cancellation between passes."""
    automaton = _chain_automaton(states=MAX_TA_STATES, rows=MAX_TA_TRANSITIONS)
    phases: list[str] = []
    monkeypatch.setattr(
        tree_operations,
        "request_checkpoint",
        lambda phase: phases.append(phase),
        raising=False,
    )

    tree_automaton_to_regular_tree_grammar(automaton)

    saturation = phases.count("during tree-automaton productivity saturation")
    assert saturation >= 1
    assert "tree automaton to regular tree grammar" in phases


def test_empty_grammar_crosses_a_cancellation_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exits of the conversion must observe cancellation."""
    automaton = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
        ),
        final_states=(0,),
    )
    assert tree_automaton_to_regular_tree_grammar(automaton).productions == ()

    def cancelled(phase: str) -> None:
        raise OperationExecutionCancelledError(phase)

    monkeypatch.setattr(tree_operations, "request_checkpoint", cancelled, raising=False)
    with pytest.raises(OperationExecutionCancelledError):
        tree_automaton_to_regular_tree_grammar(automaton)

    def expired(phase: str) -> None:
        raise OperationExecutionTimeoutError(phase)

    monkeypatch.setattr(tree_operations, "request_checkpoint", expired, raising=False)
    with pytest.raises(OperationExecutionTimeoutError):
        tree_automaton_to_regular_tree_grammar(automaton)


def test_decoded_profile_enforces_the_aggregate_witness_node_bound() -> None:
    """The bound is aggregate across states, not per witness."""
    automaton = BottomUpTreeAutomaton(
        state_count=4,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=1),
        ),
        final_states=(0, 1),
    )
    genuine = tree_language_profile(automaton)
    payload: dict[str, Any] = genuine.model_dump()
    assert len(payload["witnesses"]) >= 2

    levels = 11
    deep = RankedTree(symbol=0, children=())
    for _ in range(levels):
        deep = RankedTree(symbol=1, children=(deep, deep))
    per_witness = 2 ** (levels + 1) - 1
    assert len(payload["witnesses"]) * per_witness > MAX_REACHABILITY_WITNESS_NODES
    payload["witnesses"] = [
        {"state": row["state"], "tree": deep.model_dump()}
        for row in payload["witnesses"]
    ]

    with pytest.raises(ValueError, match="language_profile_witness_nodes"):
        TreeLanguageProfile.model_validate(payload)

    assert TreeLanguageProfile.model_validate(genuine.model_dump()) == genuine


def _unary_chain(depth: int) -> RankedTree:
    """A well-ranked chain: unary symbol 1 above a nullary symbol 0 leaf."""
    tree = RankedTree(symbol=0, children=())
    for _ in range(depth):
        tree = RankedTree(symbol=1, children=(tree,))
    return tree


def test_native_context_carrier_is_walked_without_serializing_it() -> None:
    """A deep native carrier must be refused by this boundary's own code.

    ``model_dump`` on the frame ran before the node and depth accounting, so a
    deeply nested carrier was copied first and then reported by the
    serializer's recursion guard instead of by ``tree_context.depth``.
    """
    deep = _unary_chain(MAX_RUN_TREE_DEPTH + 300)
    frame = TreeContextFrame(symbol=2, hole_child=0, siblings=(deep,))

    with pytest.raises(ValueError) as error:
        FiniteTreeContext.model_validate({"arity": [0, 1, 2], "frames": [frame]})

    assert "tree_context.depth" in str(error.value)


def test_bounded_context_carrier_is_still_admitted() -> None:
    tree = _unary_chain(4)
    frame = TreeContextFrame(symbol=2, hole_child=0, siblings=(tree,))

    revived = FiniteTreeContext.model_validate({"arity": [0, 1, 2], "frames": [frame]})

    assert revived.arity == (0, 1, 2)
    assert len(revived.frames) == 1
    assert MAX_TA_ARITY < MAX_TA_STATES
