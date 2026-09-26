from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import cast

import pytest

from jacobian.math.logic.automata.tree import (
    BottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
    complete_deterministic_tree_automaton,
    nondeterministic_run_counts,
)
from jacobian.math.logic.automata.tree import _tools as tree_tools
from jacobian.math.logic.automata.tree import values as tree_values
from jacobian.math.logic.automata.tree._models import NondeterministicRunCountsRequest
from jacobian.math.logic.automata.tree._tools import compute_nondeterministic_run_counts


def _weak_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _weak_compositions(total - first, parts - 1):
            yield (first, *rest)


def _trees_by_size(
    automaton: BottomUpTreeAutomaton, max_size: int
) -> list[list[RankedTree]]:
    by_size: list[list[RankedTree]] = [[] for _ in range(max_size + 1)]
    for size in range(1, max_size + 1):
        for symbol, arity in enumerate(automaton.arity):
            for child_sizes in _weak_compositions(size - 1, arity):
                if any(child_size == 0 for child_size in child_sizes):
                    continue
                child_lists = [by_size[child_size] for child_size in child_sizes]
                combinations = product(*child_lists) if child_lists else [()]
                by_size[size].extend(
                    RankedTree(symbol=symbol, children=children)
                    for children in combinations
                )
    return by_size


def _accepting_assignments(
    automaton: BottomUpTreeAutomaton, tree: RankedTree
) -> list[int]:
    child_assignments = [
        _accepting_assignments(automaton, child) for child in tree.children
    ]
    assignments: list[int] = []
    for child_states in product(*child_assignments) if child_assignments else [()]:
        for row in automaton.transitions:
            if row.symbol == tree.symbol and row.child_states == child_states:
                assignments.append(row.target_state)
    return assignments


def _enumeration_oracle(
    automaton: BottomUpTreeAutomaton, max_size: int
) -> tuple[int, ...]:
    final_states = set(automaton.final_states)
    result = []
    for trees in _trees_by_size(automaton, max_size)[1:]:
        result.append(
            sum(
                state in final_states
                for tree in trees
                for state in _accepting_assignments(automaton, tree)
            )
        )
    return tuple(result)


def test_run_counts_preserve_ambiguity_and_match_tree_enumeration_oracle() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=0),
        ),
        final_states=(0,),
    )

    profile = nondeterministic_run_counts(automaton, 3)

    assert profile == (1, 2, 2)
    assert profile == _enumeration_oracle(automaton, 3)


def test_deterministic_and_completed_carriers_compose_with_run_counting() -> None:
    deterministic = DeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=0),
        ),
        final_states=(0,),
    )
    completed = complete_deterministic_tree_automaton(deterministic).completed
    assert nondeterministic_run_counts(deterministic, 3) == (1, 1, 1)
    assert nondeterministic_run_counts(completed, 3) == (1, 1, 1)


def test_catalog_zero_profile_does_not_enter_polynomial_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = BottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 16),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,) * 16, target_state=0),
        ),
        final_states=(),
    )

    def cannot_enter_kernel(*_args: object) -> None:
        pytest.fail("a zero accepting-run profile must not enter the DP kernel")

    monkeypatch.setattr(
        tree_tools, "_nondeterministic_run_counts_admitted", cannot_enter_kernel
    )
    result = compute_nondeterministic_run_counts(
        NondeterministicRunCountsRequest(automaton=machine, max_size=100)
    )
    assert result.run_counts_by_size == (0,) * 100
    assert result.estimated_work_bound == 0


def test_native_and_catalog_each_saturate_ground_reachability_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = BottomUpTreeAutomaton(
        state_count=4,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            *(
                TreeAutomatonTransition(symbol=1, child_states=(i,), target_state=i + 1)
                for i in range(3)
            ),
        ),
        final_states=(3,),
    )
    calls = 0
    real_saturation = tree_values._ground_reachable_states

    def counting(value: BottomUpTreeAutomaton) -> frozenset[int]:
        nonlocal calls
        calls += 1
        return real_saturation(value)

    monkeypatch.setattr(tree_values, "_ground_reachable_states", counting)
    assert nondeterministic_run_counts(machine, 4) == (0, 0, 0, 1)
    assert calls == 1
    result = compute_nondeterministic_run_counts(
        NondeterministicRunCountsRequest(automaton=machine, max_size=4)
    )
    assert result.run_counts_by_size == (0, 0, 0, 1)
    assert result.estimated_work_bound >= 4 * machine.state_count
    assert calls == 2


def test_nullary_empty_transition_and_no_final_state_profiles() -> None:
    leaf = BottomUpTreeAutomaton(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(0,),
    )
    assert nondeterministic_run_counts(leaf, 4) == (1, 0, 0, 0)
    no_finals = BottomUpTreeAutomaton(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(),
    )
    assert nondeterministic_run_counts(no_finals, 2) == (0, 0)
    empty = BottomUpTreeAutomaton(
        state_count=1, arity=(), transitions=(), final_states=(0,)
    )
    assert nondeterministic_run_counts(empty, 3) == (0, 0, 0)


def test_result_model_preserves_zero_entries_and_source_automaton() -> None:
    automaton = BottomUpTreeAutomaton(
        state_count=1, arity=(0,), transitions=(), final_states=()
    )
    result = compute_nondeterministic_run_counts(
        NondeterministicRunCountsRequest(automaton=automaton, max_size=2)
    )

    assert result.automaton == automaton
    assert result.run_counts_by_size == (0, 0)
    assert result.estimated_work_bound >= 0


def test_zero_run_profiles_bypass_irrelevant_convolution_admission() -> None:
    no_nullary = BottomUpTreeAutomaton(
        state_count=1,
        arity=(16,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(0,) * 16, target_state=0),
        ),
        final_states=(0,),
    )
    assert nondeterministic_run_counts(no_nullary, 100) == (0,) * 100

    no_finals = BottomUpTreeAutomaton(
        state_count=1,
        arity=(16, 0),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(0,) * 16, target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=0),
        ),
        final_states=(),
    )
    assert nondeterministic_run_counts(no_finals, 100) == (0,) * 100


def test_native_entry_rejects_unvalidated_automata() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError):
        nondeterministic_run_counts(cast(BottomUpTreeAutomaton, {}), 2)
    malformed = BottomUpTreeAutomaton.model_construct(
        state_count=1, arity=(0,), transitions=(), final_states=(2,)
    )
    with pytest.raises(OperationDomainValidationError):
        nondeterministic_run_counts(malformed, 2)
