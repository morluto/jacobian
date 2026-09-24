from __future__ import annotations

from itertools import product

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree._models import (
    TreeAutomatonMinimizeRequest,
    TreeAutomatonMinimizeResult,
)
from jacobian.math.logic.automata.tree.operations import (
    minimize_tree_automaton,
    run_tree_automaton,
)
from jacobian.math.logic.automata.tree.values import (
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


def test_minimize_is_published_in_catalog() -> None:
    tool = Catalog.open().operation("tree_automaton.deterministic.minimize.compute")
    request = TreeAutomatonMinimizeRequest(
        automaton=DeterministicBottomUpTreeAutomaton(
            state_count=1,
            arity=(0,),
            transitions=(
                TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            ),
            final_states=(0,),
        )
    )

    assert tool.run(request).minimized.state_count == 1
