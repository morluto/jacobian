"""Finite-algebra conversion for complete deterministic tree automata."""

from itertools import product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.tree import (
    CompleteDeterministicBottomUpTreeAutomaton,
    RankedTree,
    TreeAutomatonTransition,
    deterministic_tree_automaton_state_algebra,
    run_tree_automaton,
)
from jacobian.math.logic.automata.tree._models import TreeRunRequest
from jacobian.math.logic.automata.tree._tools import compute_tree_run
from jacobian.math.universal_algebra.operations import evaluate_term
from jacobian.math.universal_algebra.values import (
    ApplicationTerm,
    FiniteAlgebra,
    FlatTerm,
)


def _automaton() -> CompleteDeterministicBottomUpTreeAutomaton:
    transitions = [TreeAutomatonTransition(symbol=0, child_states=(), target_state=2)]
    unary_targets = (1, 2, 0, 1)
    transitions.extend(
        TreeAutomatonTransition(symbol=1, child_states=(state,), target_state=target)
        for state, target in enumerate(unary_targets)
    )
    transitions.extend(
        TreeAutomatonTransition(
            symbol=2,
            child_states=(left, right),
            target_state=(left + 2 * right) % 3,
        )
        for left, right in product(range(4), repeat=2)
    )
    return CompleteDeterministicBottomUpTreeAutomaton(
        state_count=4,
        arity=(0, 1, 2),
        transitions=tuple(transitions),
        final_states=(2,),
    )


def _term(tree: RankedTree) -> FlatTerm:
    nodes: list[ApplicationTerm] = []
    positions: dict[int, int] = {}
    stack = [(tree, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.children))
            continue
        children = tuple(positions[id(child)] for child in node.children)
        positions[id(node)] = len(nodes)
        nodes.append(
            ApplicationTerm(
                kind="application", operation=node.symbol, children=children
            )
        )
    return FlatTerm(nodes=tuple(nodes), root=len(nodes) - 1)


def test_state_algebra_rejects_untrusted_or_incomplete_native_inputs() -> None:
    automaton = _automaton()
    partial = automaton.model_dump()
    partial["transitions"] = partial["transitions"][:-1]
    with pytest.raises(OperationDomainValidationError):
        deterministic_tree_automaton_state_algebra(partial)
    with pytest.raises(OperationDomainValidationError):
        deterministic_tree_automaton_state_algebra(object())  # type: ignore[arg-type]
    malformed = CompleteDeterministicBottomUpTreeAutomaton.model_construct(**partial)
    with pytest.raises(OperationDomainValidationError):
        deterministic_tree_automaton_state_algebra(malformed)


def test_state_algebra_preserves_full_state_and_ranked_symbol_axes() -> None:
    automaton = _automaton()

    algebra = deterministic_tree_automaton_state_algebra(automaton)

    assert algebra.carrier == ("state_0", "state_1", "state_2", "state_3")
    assert tuple((op.operation_id, op.arity) for op in algebra.operations) == (
        ("tree_symbol_0", 0),
        ("tree_symbol_1", 1),
        ("tree_symbol_2", 2),
    )
    assert algebra.tables == (
        (2,),
        (1, 2, 0, 1),
        tuple((left + 2 * right) % 3 for left, right in product(range(4), repeat=2)),
    )
    assert (
        "state_3" in algebra.carrier
    )  # state 3 is unreachable from every ground tree.


def test_algebra_term_evaluation_agrees_with_automaton_runs_after_json_roundtrip() -> (
    None
):
    automaton = _automaton()
    algebra = deterministic_tree_automaton_state_algebra(automaton)
    decoded = FiniteAlgebra.model_validate_json(algebra.model_dump_json())
    trees = (
        RankedTree(symbol=0, children=()),
        RankedTree(
            symbol=1,
            children=(RankedTree(symbol=0, children=()),),
        ),
        RankedTree(
            symbol=2,
            children=(
                RankedTree(symbol=0, children=()),
                RankedTree(symbol=1, children=(RankedTree(symbol=0, children=()),)),
            ),
        ),
    )

    for tree in trees:
        root_states = run_tree_automaton(automaton, tree)
        run = compute_tree_run(TreeRunRequest(automaton=automaton, tree=tree))
        algebra_state = evaluate_term(decoded, _term(tree), {})
        assert root_states == {algebra_state}
        assert run.root_states == (algebra_state,)
        assert run.accepted is (algebra_state in automaton.final_states)


def test_state_algebra_bounds_carrier_before_table_expansion() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=33,
        arity=(),
        transitions=(),
        final_states=(),
    )

    with pytest.raises(OperationResourceAdmissionError, match="carrier bound"):
        deterministic_tree_automaton_state_algebra(automaton)


def test_empty_ranked_alphabet_preserves_carrier_without_operations() -> None:
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=1,
        arity=(),
        transitions=(),
        final_states=(),
    )

    algebra = deterministic_tree_automaton_state_algebra(automaton)

    assert algebra.carrier == ("state_0",)
    assert algebra.operations == ()
    assert algebra.tables == ()


def test_state_algebra_rejects_rank_exceeding_universal_algebra_bound() -> None:
    transitions = tuple(
        TreeAutomatonTransition(
            symbol=0,
            child_states=children,
            target_state=sum(children) % 2,
        )
        for children in product(range(2), repeat=5)
    )
    automaton = CompleteDeterministicBottomUpTreeAutomaton(
        state_count=2,
        arity=(5,),
        transitions=transitions,
        final_states=(1,),
    )

    with pytest.raises(OperationResourceAdmissionError, match="arity"):
        deterministic_tree_automaton_state_algebra(automaton)
