from __future__ import annotations

import time
from itertools import product
from typing import Any

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    request_cancellation,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree._models import (
    TreeAutomatonMinimizeResult,
)
from jacobian.math.logic.automata.tree.operations import (
    complement_tree_automaton,
    minimize_tree_automaton,
    run_tree_automaton,
)
from jacobian.math.logic.automata.tree.values import (
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
)


def _small_trees(arity: tuple[int, ...], max_size: int) -> tuple[RankedTree, ...]:
    by_size: dict[int, tuple[RankedTree, ...]] = {}
    for symbol, rank in enumerate(arity):
        if rank == 0:
            by_size.setdefault(1, ())
            by_size[1] += (RankedTree(symbol=symbol),)
    for size in range(2, max_size + 1):
        trees: list[RankedTree] = []
        for symbol, rank in enumerate(arity):
            if rank == 0:
                continue
            for child_sizes in product(range(1, size), repeat=rank):
                if 1 + sum(child_sizes) != size:
                    continue
                child_groups = (by_size[child_size] for child_size in child_sizes)
                for children in product(*child_groups):
                    trees.append(RankedTree(symbol=symbol, children=children))
        by_size[size] = tuple(trees)
    return tuple(
        tree for size in range(1, max_size + 1) for tree in by_size.get(size, ())
    )


def _accepted(machine: DeterministicBottomUpTreeAutomaton, tree: RankedTree) -> bool:
    return bool(set(run_tree_automaton(machine, tree)) & set(machine.final_states))


def test_minimize_merges_equivalent_reachable_states_and_drops_unreachable() -> None:
    rows = tuple(
        TreeAutomatonTransition(
            symbol=symbol, child_states=children, target_state=target
        )
        for symbol, rank in enumerate((0, 1))
        for children in product(range(4), repeat=rank)
        for target in (
            0 if symbol == 0 else (1 if children[0] in (0, 1) else children[0]),
        )
    )
    source = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=4,
        arity=(0, 1),
        transitions=rows,
        final_states=(),
    )

    result = minimize_tree_automaton(source)

    assert isinstance(result, TreeAutomatonMinimizeResult)
    assert result.reachable_states == (0, 1)
    assert result.old_to_new == (0, 0, -1, -1)
    assert result.new_to_old == (0,)
    assert result.minimized.state_count == 1
    CompleteDeterministicBottomUpTreeAutomaton.model_validate(
        result.minimized.model_dump(), strict=True
    )
    assert len(result.minimized.transitions) == 2
    for tree in _small_trees(source.arity, 5):
        assert _accepted(source, tree) == _accepted(result.minimized, tree)
    round_trip = TreeAutomatonMinimizeResult.model_validate_json(
        result.model_dump_json()
    )
    assert round_trip.model_dump() == result.model_dump()


def test_sink_equivalent_block_uses_real_state_order_and_roundtrips() -> None:
    machine = DeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 0),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
        ),
        final_states=(0,),
    )
    result = minimize_tree_automaton(machine)
    assert result.new_to_old == (0, 1)
    assert result.old_to_new == (0, 1)
    assert run_tree_automaton(result.minimized, RankedTree(symbol=0)) == {0}
    assert run_tree_automaton(result.minimized, RankedTree(symbol=1)) == {1}
    assert (
        TreeAutomatonMinimizeResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_minimize_preserves_partial_undefined_transitions() -> None:
    source = DeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
        ),
        final_states=(1,),
    )

    result = minimize_tree_automaton(source)

    assert result.old_to_new == (0, 1)
    assert result.minimized.transitions == source.transitions
    assert result.minimized.final_states == source.final_states
    assert _accepted(source, RankedTree(symbol=1, children=(RankedTree(symbol=0),)))
    assert not _accepted(
        result.minimized,
        RankedTree(
            symbol=1,
            children=(RankedTree(symbol=1, children=(RankedTree(symbol=0),)),),
        ),
    )


def test_minimize_empty_ground_tree_language_has_canonical_carrier() -> None:
    source = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=3,
        arity=(1,),
        transitions=tuple(
            TreeAutomatonTransition(symbol=0, child_states=(state,), target_state=state)
            for state in range(3)
        ),
        final_states=(1,),
    )

    result = minimize_tree_automaton(source)

    assert result.reachable_states == ()
    assert result.old_to_new == (-1, -1, -1)
    assert result.new_to_old == (None,)
    assert result.minimized.state_count == 1
    assert result.minimized.transitions == ()
    assert result.minimized.final_states == ()
    assert (
        TreeAutomatonMinimizeResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_minimize_admits_partition_work_before_refinement() -> None:
    source = DeterministicBottomUpTreeAutomaton(
        state_count=64,
        arity=(0, 16),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(),
    )

    with pytest.raises(OperationResourceAdmissionError):
        minimize_tree_automaton(source)


def test_minimize_translates_forged_carrier_shape_errors() -> None:
    forged = DeterministicBottomUpTreeAutomaton.model_construct(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=4),
        ),
        final_states=(),
    )

    with pytest.raises(OperationDomainValidationError) as raised:
        minimize_tree_automaton(forged)

    assert raised.value.errors()[0]["type"] == "tree_automata.minimize_automaton_shape"


def _stuck_rejects(machine: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
    table = {
        (row.symbol, row.child_states): row.target_state for row in machine.transitions
    }

    def state(node: RankedTree) -> int | None:
        children = tuple(state(child) for child in node.children)
        if any(child is None for child in children):
            return None
        return table.get(
            (node.symbol, tuple(child for child in children if child is not None))
        )

    root = state(tree)
    return root is not None and root in machine.final_states


def test_minimize_merges_states_that_reject_in_every_ground_tree_context() -> None:
    # a -> 0, b -> 1, f(0) -> 0, and no final states: both reachable states
    # reject in every ground-tree context, so the smallest contextual quotient
    # is one state. A missing row must behave like the implicit rejecting sink.
    source = DeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(0,), target_state=0),
        ),
        final_states=(),
    )

    result = minimize_tree_automaton(source)

    assert result.reachable_states == (0, 1)
    assert result.old_to_new == (0, 0)
    assert result.minimized.state_count == 1
    for tree in _small_trees(source.arity, 5):
        assert _stuck_rejects(source, tree) == _stuck_rejects(result.minimized, tree)


def _reachable_by_ground_tree(
    machine: BottomUpTreeAutomaton,
) -> tuple[int, ...]:
    reachable: set[int] = set()
    changed = True
    while changed:
        changed = False
        for row in machine.transitions:
            if all(state in reachable for state in row.child_states) and (
                row.target_state not in reachable
            ):
                reachable.add(row.target_state)
                changed = True
    return tuple(sorted(reachable))


def _encoded(tree: RankedTree) -> Any:
    return (tree.symbol, tuple(_encoded(child) for child in tree.children))


def _hole_contexts(arity: tuple[int, ...]) -> list[Any]:
    ground = [_encoded(tree) for tree in _small_trees(arity, 4)]
    contexts: list[Any] = ["H"]
    for _ in range(2):
        extended: list[Any] = list(contexts)
        for context in contexts:
            for symbol, rank in enumerate(arity):
                if rank == 0:
                    continue
                for hole_side in range(rank):
                    for fillers in product(ground, repeat=rank - 1):
                        children = list[Any](fillers)
                        children.insert(hole_side, context)
                        extended.append((symbol, tuple(children)))
        contexts = extended
    return contexts


def _context_acceptance(
    machine: BottomUpTreeAutomaton, start: int, contexts: list[Any]
) -> tuple[bool, ...]:
    table = {
        (row.symbol, row.child_states): row.target_state for row in machine.transitions
    }
    finals = set(machine.final_states)

    def evaluate(node: Any) -> int | None:
        if node == "H":
            return start
        children = tuple(evaluate(child) for child in node[1])
        if any(child is None for child in children):
            return None
        return table.get(
            (node[0], tuple(child for child in children if child is not None))
        )

    return tuple(evaluate(context) in finals for context in contexts)


def test_minimize_matches_exhaustive_contextual_oracle_on_small_machines() -> None:
    # Sweep every partial two-state table over a nullary pair and one unary
    # symbol, times every final subset, and compare the quotient against
    # exhaustive acceptance over all bounded ground-tree hole contexts.
    arity = (0, 0, 1)
    keys = ((0, ()), (1, ()), (2, (0,)), (2, (1,)))
    contexts = _hole_contexts(arity)
    for choice in product((None, 0, 1), repeat=len(keys)):
        rows = tuple(
            TreeAutomatonTransition(
                symbol=symbol, child_states=children, target_state=target
            )
            for (symbol, children), target in zip(keys, choice, strict=True)
            if target is not None
        )
        for finals in ((), (0,), (1,), (0, 1)):
            machine = DeterministicBottomUpTreeAutomaton(
                state_count=2,
                arity=arity,
                transitions=rows,
                final_states=finals,
            )
            result = minimize_tree_automaton(machine)
            reachable = _reachable_by_ground_tree(machine)
            if not reachable:
                assert result.old_to_new == (-1, -1)
                assert result.minimized.state_count == 1
                continue
            vectors = {
                state: _context_acceptance(machine, state, contexts)
                for state in reachable
            }
            distinct = {vectors[state] for state in reachable}
            assert result.minimized.state_count == len(distinct), (rows, finals)
            for left in reachable:
                for right in reachable:
                    same_class = result.old_to_new[left] == result.old_to_new[right]
                    assert same_class == (vectors[left] == vectors[right]), (
                        rows,
                        finals,
                        left,
                        right,
                    )
            for tree in _small_trees(arity, 5):
                assert _stuck_rejects(machine, tree) == _stuck_rejects(
                    result.minimized, tree
                ), (rows, finals, tree)


def test_minimized_complete_quotient_keeps_the_complete_carrier() -> None:
    source = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=0),
        ),
        final_states=(0,),
    )

    result = minimize_tree_automaton(source)

    assert isinstance(result.minimized, CompleteDeterministicBottomUpTreeAutomaton)
    decoded = TreeAutomatonMinimizeResult.model_validate_json(result.model_dump_json())
    assert isinstance(decoded.minimized, CompleteDeterministicBottomUpTreeAutomaton)
    complemented = complement_tree_automaton(result.minimized)
    for tree in _small_trees(source.arity, 5):
        assert _stuck_rejects(source, tree) == _stuck_rejects(result.minimized, tree)
        assert not _stuck_rejects(complemented.complement, tree)


def test_minimized_partial_quotient_keeps_the_partial_carrier() -> None:
    source = DeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
        ),
        final_states=(1,),
    )

    result = minimize_tree_automaton(source)

    assert not isinstance(result.minimized, CompleteDeterministicBottomUpTreeAutomaton)
    decoded = TreeAutomatonMinimizeResult.model_validate_json(result.model_dump_json())
    assert not isinstance(decoded.minimized, CompleteDeterministicBottomUpTreeAutomaton)


class _CountingCancellation:
    def __init__(self, fire_on: int) -> None:
        self.fire_on = fire_on
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks >= self.fire_on


def test_minimize_partition_refinement_honours_cancellation() -> None:
    # A 64-state unary chain separates one state per refinement round. The
    # refinement loop must consult the cancellation signal, not run to
    # completion after the client has cancelled.
    source = DeterministicBottomUpTreeAutomaton(
        state_count=64,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            *(
                TreeAutomatonTransition(
                    symbol=1, child_states=(state,), target_state=state + 1
                )
                for state in range(63)
            ),
        ),
        final_states=(63,),
    )
    signal = _CountingCancellation(fire_on=8)

    with (
        request_execution(started_at=time.monotonic()),
        request_cancellation(signal),
        pytest.raises(OperationExecutionCancelledError, match="partition refinement"),
    ):
        minimize_tree_automaton(source)

    assert signal.checks == 8
