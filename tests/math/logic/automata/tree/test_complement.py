"""Complement truth tables over bounded finite ranked tree universes."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.tree import (
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonComplementResult,
    TreeAutomatonTransition,
    complement_tree_automaton,
)


def _parity_automaton() -> CompleteDeterministicBottomUpTreeAutomaton:
    # Two leaf symbols seed opposite states; the binary symbol computes XOR.
    # The first row also forces a nonidentity canonical state relabeling.
    return CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 0, 2),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 0), target_state=0),
            TreeAutomatonTransition(symbol=2, child_states=(0, 1), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(1, 0), target_state=1),
            TreeAutomatonTransition(symbol=2, child_states=(1, 1), target_state=0),
        ),
        final_states=(1,),
    )


def _trees_at_height_most(
    automaton: BottomUpTreeAutomaton, height: int
) -> tuple[RankedTree, ...]:
    by_height: list[tuple[RankedTree, ...]] = []
    trees: set[RankedTree] = set()
    for symbol, arity in enumerate(automaton.arity):
        if arity == 0:
            trees.add(RankedTree(symbol=symbol))
    by_height.append(tuple(sorted(trees, key=lambda tree: tree.symbol)))
    for _ in range(1, height):
        prior = tuple(trees)
        for symbol, arity in enumerate(automaton.arity):
            if arity:
                trees.update(
                    RankedTree(symbol=symbol, children=children)
                    for children in product(prior, repeat=arity)
                )
        by_height.append(tuple(trees))
    return tuple(trees)


def _accepted(automaton: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
    # Independent direct recursive evaluation of the deterministic transition
    # table; deliberately does not call the production run operation.
    table = {
        (row.symbol, row.child_states): row.target_state
        for row in automaton.transitions
    }

    def state(node: RankedTree) -> int:
        return table[(node.symbol, tuple(state(child) for child in node.children))]

    return state(tree) in automaton.final_states


def test_complement_flips_every_tree_in_a_finite_truth_table() -> None:
    source = _parity_automaton()
    result = complement_tree_automaton(source)
    assert isinstance(result, TreeAutomatonComplementResult)
    assert result.old_to_new == (1, 0)
    assert result.new_to_old == (1, 0)
    assert result.transition_count == 6
    trees = _trees_at_height_most(source, 3)
    assert len(trees) == 38
    assert all(
        _accepted(result.complement, tree) is not _accepted(source, tree)
        for tree in trees
    )
    assert (
        TreeAutomatonComplementResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_complement_rejects_incomplete_transition_tables() -> None:
    incomplete = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=0),
        ),
        final_states=(0,),
    )
    with pytest.raises(OperationDomainValidationError, match="complete deterministic"):
        complement_tree_automaton(incomplete)  # type: ignore[arg-type]


def test_complement_rejects_nondeterministic_transition_keys() -> None:
    nondeterministic = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
        ),
        final_states=(0,),
    )
    with pytest.raises(OperationDomainValidationError, match="complete deterministic"):
        complement_tree_automaton(nondeterministic)  # type: ignore[arg-type]


def test_complement_admits_transition_product_before_expansion() -> None:
    high_arity = DeterministicBottomUpTreeAutomaton(
        state_count=64,
        arity=(3,),
        transitions=(),
        final_states=(),
    )
    with pytest.raises(ValidationError, match="one transition for every symbol"):
        CompleteDeterministicBottomUpTreeAutomaton.model_validate(
            high_arity.model_dump(), strict=True
        )


def test_catalog_complement_operation_uses_the_typed_contract() -> None:
    tool = Catalog.open().operation("tree_automaton.complement.compute")
    assert tool is not None
    request = tool.request_type.model_validate(
        {
            "automaton": {
                "state_count": 2,
                "arity": [0],
                "transitions": [{"symbol": 0, "child_states": [], "target_state": 1}],
                "final_states": [1],
            }
        }
    )
    result = tool.run(request)
    assert result.complement.final_states == (1,)
