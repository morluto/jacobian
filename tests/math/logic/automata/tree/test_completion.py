"""Independent finite-language checks for deterministic completion."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.tree import (
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
    complete_deterministic_tree_automaton,
)


def _partial_accepts(automaton: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
    """Evaluate a partial deterministic run; a stuck run rejects."""
    table = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }

    def state(node: RankedTree) -> int | None:
        child_states = tuple(state(child) for child in node.children)
        if any(child is None for child in child_states):
            return None
        return table.get(
            (node.symbol, tuple(child for child in child_states if child is not None))
        )

    result = state(tree)
    return result is not None and result in automaton.final_states


def _completed_accepts(automaton: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
    """Evaluate a complete deterministic table without production kernels."""
    table = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }

    def state(node: RankedTree) -> int:
        return table[(node.symbol, tuple(state(child) for child in node.children))]

    return state(tree) in automaton.final_states


def _trees_through_height(max_height: int) -> tuple[RankedTree, ...]:
    levels: list[tuple[RankedTree, ...]] = [(RankedTree(symbol=0),)]
    all_trees = {levels[0][0]}
    for _ in range(1, max_height):
        previous = tuple(all_trees)
        new_trees = {
            RankedTree(symbol=0),
            *(
                RankedTree(symbol=1, children=children)
                for children in product(previous, repeat=2)
            ),
        }
        all_trees.update(new_trees)
        levels.append(tuple(new_trees))
    return tuple(all_trees)


def test_completion_preserves_partial_language_for_every_small_table() -> None:
    # A constant and a binary symbol over two states have five deterministic
    # keys. Exhaust every partial domain and every final-state subset; assign
    # both target values across the family of tables.
    keys = (
        (0, ()),
        *((1, children) for children in product(range(2), repeat=2)),
    )
    trees = _trees_through_height(4)
    assert len(trees) == 26
    for present_mask in range(1 << len(keys)):
        rows = tuple(
            TreeAutomatonTransition(
                symbol=symbol,
                child_states=children,
                target_state=(index + present_mask).bit_count() % 2,
            )
            for index, (symbol, children) in enumerate(keys)
            if present_mask & (1 << index)
        )
        for finals in ((), (0,), (1,), (0, 1)):
            source = DeterministicBottomUpTreeAutomaton(
                state_count=2,
                arity=(0, 2),
                transitions=rows,
                final_states=finals,
            )
            result = complete_deterministic_tree_automaton(source)
            assert result.completed.state_count == (2 if len(rows) == len(keys) else 3)
            assert result.sink_state == (None if len(rows) == len(keys) else 2)
            assert all(
                _partial_accepts(source, tree)
                == _completed_accepts(result.completed, tree)
                for tree in trees
            )


def test_complete_input_is_preserved_without_adding_a_state() -> None:
    source = DeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
        ),
        final_states=(0,),
    )
    result = complete_deterministic_tree_automaton(source)
    assert result.sink_state is None
    assert result.completed.state_count == 1
    assert result.completed.transitions == source.transitions
    assert isinstance(result.completed, CompleteDeterministicBottomUpTreeAutomaton)
    assert result.source_to_completed == (0,)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_deterministic_carrier_rejects_ambiguous_transition_keys() -> None:
    with pytest.raises(ValidationError, match="at most one target"):
        DeterministicBottomUpTreeAutomaton(
            state_count=2,
            arity=(0,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            ),
            final_states=(0,),
        )


def test_completion_rejects_sink_or_output_that_exceeds_bounds() -> None:
    no_room_for_sink = DeterministicBottomUpTreeAutomaton(
        state_count=64, arity=(0,), transitions=(), final_states=()
    )
    with pytest.raises(OperationResourceAdmissionError, match="one sink state"):
        complete_deterministic_tree_automaton(no_room_for_sink)

    expanded_table_too_large = DeterministicBottomUpTreeAutomaton(
        state_count=16, arity=(3,), transitions=(), final_states=()
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="completed transition table"
    ):
        complete_deterministic_tree_automaton(expanded_table_too_large)


def test_catalog_completion_operation_has_typed_json_contract() -> None:
    operation = Catalog.open().operation(
        "tree_automaton.deterministic.complete.compute"
    )
    assert operation is not None
    request = operation.request_type.model_validate(
        {
            "automaton": {
                "state_count": 1,
                "arity": [0, 1],
                "transitions": [{"symbol": 0, "child_states": [], "target_state": 0}],
                "final_states": [0],
            }
        }
    )
    result = operation.run(request)
    assert result.completed.state_count == 2
    assert result.sink_state == 1
