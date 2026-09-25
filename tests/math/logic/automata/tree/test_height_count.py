"""Independent tiny-tree oracle for height-bounded language counts."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Any

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree import (
    AcceptedTreeHeightProfileRequest,
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    TreeAutomatonTransition,
    accepted_tree_height_profile,
)
from jacobian.math.logic.automata.tree._tools import (
    compute_accepted_tree_height_profile,
)

Tree = tuple[int, tuple["Tree", ...]]


def _fixture() -> CompleteDeterministicBottomUpTreeAutomaton:
    return CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(0, 1), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(1, 0), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=1),
        ),
        final_states=(1,),
    )


def _oracle(automaton: BottomUpTreeAutomaton, max_height: int) -> tuple[int, ...]:
    transitions = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }
    cumulative: list[tuple[Tree, int]] = []
    profile = []
    for height in range(max_height + 1):
        new_trees: list[Tree] = []
        if height == 0:
            new_trees.extend(
                (symbol, ()) for symbol, rank in enumerate(automaton.arity) if rank == 0
            )
        else:
            for symbol, rank in enumerate(automaton.arity):
                if not rank:
                    continue
                child_families = [cumulative] * rank
                for children_with_height in product(*child_families):
                    if (
                        max(child_height for _, child_height in children_with_height)
                        != height - 1
                    ):
                        continue
                    new_trees.append(
                        (
                            symbol,
                            tuple(tree for tree, _ in children_with_height),
                        )
                    )
        cumulative.extend((tree, height) for tree in new_trees)

        def evaluate(tree: Tree) -> int:
            symbol, children = tree
            child_states = tuple(evaluate(child) for child in children)
            return transitions[(symbol, child_states)]

        profile.append(
            sum(evaluate(tree) in automaton.final_states for tree, _ in cumulative)
        )
    return tuple(profile)


def test_no_nullary_symbols_short_circuit_work_bound() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(12,),
        transitions=tuple(
            TreeAutomatonTransition(symbol=0, child_states=children, target_state=0)
            for children in product(range(2), repeat=12)
        ),
        final_states=(0,),
    )
    assert accepted_tree_height_profile(automaton, 100) == (0,) * 101


def test_height_profile_matches_complete_tiny_tree_enumeration() -> None:
    automaton = _fixture()
    expected = _oracle(automaton, 3)

    assert accepted_tree_height_profile(automaton, 3) == expected
    result = compute_accepted_tree_height_profile(
        AcceptedTreeHeightProfileRequest(automaton=automaton, max_height=3)
    )
    assert result.counts_by_height == expected
    assert result.max_height == 3
    assert (
        CompleteDeterministicBottomUpTreeAutomaton.model_validate_json(
            automaton.model_dump_json()
        )
        == automaton
    )


def test_height_zero_counts_only_nullary_symbols() -> None:
    automaton = _fixture()
    assert accepted_tree_height_profile(automaton, 0) == (0,)


def test_no_nullary_symbols_short_circuit_transition_work() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(12,),
        transitions=tuple(
            TreeAutomatonTransition(symbol=0, child_states=children, target_state=0)
            for children in product(range(2), repeat=12)
        ),
        final_states=(0,),
    )
    assert accepted_tree_height_profile(automaton, 100) == (0,) * 101


def test_empty_final_set_has_zero_profile_without_integer_growth() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
        ),
        final_states=(),
    )
    assert accepted_tree_height_profile(automaton, 100) == (0,) * 101


def test_integer_growth_is_admitted_before_recurrence() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0, 0), target_state=0),
        ),
        final_states=(0,),
    )
    with pytest.raises(OperationResourceAdmissionError, match="all-trees upper bound"):
        accepted_tree_height_profile(automaton, 19)


def test_out_of_range_height_is_a_resource_refusal() -> None:
    automaton = _fixture()
    for invalid in (-1, 101):
        with pytest.raises(OperationResourceAdmissionError):
            accepted_tree_height_profile(automaton, invalid)


@pytest.mark.parametrize("invalid", [False, True, 3.0, Fraction(3, 1), None])
def test_non_integer_height_is_a_domain_error(invalid: Any) -> None:
    automaton = _fixture()
    with pytest.raises(OperationDomainValidationError, match="integer"):
        accepted_tree_height_profile(automaton, invalid)
