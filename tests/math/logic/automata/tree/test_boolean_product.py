"""Independent language oracle for bounded tree-automaton Boolean products."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationMatchRequest,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree import (
    BottomUpTreeAutomaton,
    CompleteDeterministicBottomUpTreeAutomaton,
    DeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
    boolean_product_tree_automata,
)
from jacobian.math.logic.automata.tree._models import TreeAutomatonBooleanProductRequest


def _trees(arity: tuple[int, ...], height: int) -> list[RankedTree]:
    levels: list[list[RankedTree]] = [[]]
    all_trees: list[RankedTree] = []
    for _ in range(height + 1):
        current: list[RankedTree] = []
        for symbol, rank in enumerate(arity):
            if rank == 0:
                current.append(RankedTree(symbol=symbol))
            else:
                import itertools

                current.extend(
                    RankedTree(symbol=symbol, children=children)
                    for children in itertools.product(levels[-1], repeat=rank)
                )
        levels.append(current)
        all_trees.extend(current)
    return all_trees


def _accepts(machine: BottomUpTreeAutomaton, tree: RankedTree) -> bool:
    rows = {
        (row.symbol, row.child_states): row.target_state for row in machine.transitions
    }

    def state(node: RankedTree):
        children = tuple(state(child) for child in node.children)
        if any(child is None for child in children):
            return None
        return rows.get((node.symbol, children))

    root = state(tree)
    return root is not None and root in machine.final_states


def _machine(finals: tuple[int, ...]) -> CompleteDeterministicBottomUpTreeAutomaton:
    return CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=1, child_states=(0,), target_state=1),
            TreeAutomatonTransition(symbol=1, child_states=(1,), target_state=0),
        ),
        final_states=finals,
    )


@pytest.mark.parametrize(
    "connective", ["intersection", "union", "difference", "symmetric_difference"]
)
def test_boolean_product_agrees_with_independent_tree_evaluation(
    connective: str,
) -> None:
    left, right = _machine((0,)), _machine((1,))
    result = boolean_product_tree_automata(left, right, connective)  # type: ignore[arg-type]
    assert len(result.state_pairs) == 4
    for tree in _trees(left.arity, 6):
        lhs, rhs = _accepts(left, tree), _accepts(right, tree)
        expected = {
            "intersection": lhs and rhs,
            "union": lhs or rhs,
            "difference": lhs and not rhs,
            "symmetric_difference": lhs != rhs,
        }[connective]
        assert _accepts(result.product, tree) == expected
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result


def test_partial_input_is_rejected_instead_of_claiming_union() -> None:
    left = _machine((0,))
    right = DeterministicBottomUpTreeAutomaton(
        state_count=1, arity=(0, 1), transitions=(), final_states=()
    )
    with pytest.raises(ValidationError):
        TreeAutomatonBooleanProductRequest(
            left=left,
            right=right,
            connective="union",
        )


def test_binary_transition_children_are_paired_positionally() -> None:
    rows = [TreeAutomatonTransition(symbol=0, child_states=(), target_state=0)]
    rows.extend(
        TreeAutomatonTransition(symbol=1, child_states=(state,), target_state=1 - state)
        for state in range(2)
    )
    rows.extend(
        TreeAutomatonTransition(
            symbol=2, child_states=(left, right), target_state=left ^ right
        )
        for left in range(2)
        for right in range(2)
    )
    left = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2, arity=(0, 1, 2), transitions=tuple(rows), final_states=(1,)
    )
    right_rows = tuple(
        TreeAutomatonTransition(
            symbol=row.symbol,
            child_states=row.child_states,
            target_state=(
                row.target_state if row.symbol != 2 else 1 - row.target_state
            ),
        )
        for row in rows
    )
    right = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2, arity=(0, 1, 2), transitions=right_rows, final_states=(1,)
    )
    result = boolean_product_tree_automata(left, right, "symmetric_difference")
    for tree in _trees(left.arity, 2):
        assert _accepts(result.product, tree) == (
            _accepts(left, tree) != _accepts(right, tree)
        )


def test_rejects_nondeterministic_input_and_mismatched_signature() -> None:
    left = _machine((0,))
    nondeterministic = BottomUpTreeAutomaton(
        state_count=2,
        arity=(0, 1),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=1),
        ),
        final_states=(0,),
    )
    with pytest.raises(ValidationError, match="valid dictionary or instance"):
        TreeAutomatonBooleanProductRequest(
            left=nondeterministic,
            right=left,
            connective="intersection",
        )
    other_signature = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(0,),
        transitions=(
            TreeAutomatonTransition(symbol=0, child_states=(), target_state=0),
        ),
        final_states=(0,),
    )
    with pytest.raises(OperationDomainValidationError, match="ranked signature"):
        boolean_product_tree_automata(left, other_signature, "intersection")


def test_pair_state_product_is_admitted_before_expansion() -> None:
    machine = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=9, arity=(), transitions=(), final_states=()
    )
    with pytest.raises(OperationResourceAdmissionError, match="state pairs"):
        boolean_product_tree_automata(machine, machine, "intersection")


def test_invalid_connective_rejected_by_typed_request() -> None:
    with pytest.raises(ValidationError):
        TreeAutomatonBooleanProductRequest.model_validate(
            {
                "left": _machine(()),
                "right": _machine(()),
                "connective": "xor-ish",
            }
        )


def test_catalog_discovery_surfaces_tree_language_products() -> None:
    result = Catalog.open().match(
        OperationMatchRequest(
            need="exact intersection and union product for deterministic ranked tree automata",
            limit=5,
        )
    )
    assert result.matches[0].operation_id == "tree_automaton.boolean_product.compute"
