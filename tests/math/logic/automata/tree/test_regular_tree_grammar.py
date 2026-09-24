from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.logic.automata.tree._models import (
    RegularTreeGrammarToAutomatonRequest,
    RegularTreeGrammarToAutomatonResult,
)
from jacobian.math.logic.automata.tree._tools import TOOLS
from jacobian.math.logic.automata.tree.operations import (
    regular_tree_grammar_to_automaton,
    run_tree_automaton,
)
from jacobian.math.logic.automata.tree.values import (
    RankedTree,
    RegularTreeGrammar,
    RegularTreeProduction,
)


def _grammar() -> RegularTreeGrammar:
    # S -> f(B,A), A -> c | f(A,A), B -> u(A).
    return RegularTreeGrammar(
        nonterminal_count=3,
        arity=(0, 1, 2),
        start_nonterminal=0,
        productions=(
            RegularTreeProduction(nonterminal=0, symbol=2, children=(2, 1)),
            RegularTreeProduction(nonterminal=1, symbol=0, children=()),
            RegularTreeProduction(nonterminal=1, symbol=2, children=(1, 1)),
            RegularTreeProduction(nonterminal=2, symbol=1, children=(1,)),
        ),
    )


def _trees_by_size(max_size: int) -> tuple[RankedTree, ...]:
    by_size: dict[int, list[RankedTree]] = {1: [RankedTree(symbol=0)]}
    for size in range(2, max_size + 1):
        trees: list[RankedTree] = []
        for symbol, rank in enumerate((0, 1, 2)):
            if rank == 0:
                continue
            for first_size in range(1, size):
                child_sizes = (
                    ((first_size,),)
                    if rank == 1
                    else tuple(
                        (first_size, second_size)
                        for second_size in range(1, size - first_size)
                    )
                )
                for sizes in child_sizes:
                    if sum(sizes) != size - 1 or any(
                        value not in by_size for value in sizes
                    ):
                        continue
                    for children in product(*(by_size[value] for value in sizes)):
                        trees.append(RankedTree(symbol=symbol, children=children))
        by_size[size] = trees
    return tuple(tree for size in sorted(by_size) for tree in by_size[size])


def _grammar_root_states(
    grammar: RegularTreeGrammar, tree: RankedTree
) -> frozenset[int]:
    child_states = tuple(
        _grammar_root_states(grammar, child) for child in tree.children
    )
    result = set()
    for production in grammar.productions:
        if production.symbol != tree.symbol:
            continue
        if len(production.children) == len(child_states) and all(
            state in possible
            for state, possible in zip(production.children, child_states, strict=True)
        ):
            result.add(production.nonterminal)
    return frozenset(result)


def test_conversion_preserves_grammar_derivations_as_automaton_runs() -> None:
    grammar = _grammar()
    result = regular_tree_grammar_to_automaton(
        RegularTreeGrammarToAutomatonRequest(grammar=grammar)
    )

    assert result.automaton.state_count == grammar.nonterminal_count
    assert result.automaton.arity == grammar.arity
    assert result.automaton.final_states == (grammar.start_nonterminal,)
    assert result.grammar == grammar
    for tree in _trees_by_size(5):
        grammar_states = _grammar_root_states(grammar, tree)
        automaton_states = tuple(sorted(run_tree_automaton(result.automaton, tree)))
        assert automaton_states == tuple(sorted(grammar_states))
        assert (grammar.start_nonterminal in automaton_states) == (
            grammar.start_nonterminal in grammar_states
        )
    assert (
        RegularTreeGrammarToAutomatonResult.model_validate(result.model_dump())
        == result
    )


def test_production_input_order_is_transport_only() -> None:
    grammar = _grammar()
    raw = grammar.model_dump()
    raw["productions"] = list(reversed(raw["productions"]))

    assert RegularTreeGrammar.model_validate(raw) == grammar


def test_empty_grammar_preserves_empty_signature_and_language() -> None:
    grammar = RegularTreeGrammar(
        nonterminal_count=2,
        arity=(),
        start_nonterminal=1,
        productions=(),
    )

    result = regular_tree_grammar_to_automaton(
        RegularTreeGrammarToAutomatonRequest(grammar=grammar)
    )

    assert result.automaton.arity == ()
    assert result.automaton.transitions == ()
    assert result.automaton.final_states == (1,)
    assert result.grammar == grammar


def test_maximum_admitted_rule_and_rank_shape_converts_within_work_bound() -> None:
    productions = tuple(
        RegularTreeProduction(
            nonterminal=nonterminal,
            symbol=0,
            children=(first_child,) + (0,) * 15,
        )
        for nonterminal in range(64)
        for first_child in range(64)
    )
    grammar = RegularTreeGrammar(
        nonterminal_count=64,
        arity=(16,),
        start_nonterminal=0,
        productions=productions,
    )

    result = regular_tree_grammar_to_automaton(
        RegularTreeGrammarToAutomatonRequest(grammar=grammar)
    )

    assert len(result.automaton.transitions) == 4096
    assert result.automaton.transitions[-1].child_states == (63,) + (0,) * 15


@pytest.mark.parametrize(
    "payload",
    [
        {
            "nonterminal_count": 1,
            "arity": [0],
            "start_nonterminal": 0,
            "productions": [
                {"nonterminal": 0, "symbol": 0, "children": []},
                {"nonterminal": 0, "symbol": 0, "children": []},
            ],
        },
        {
            "nonterminal_count": 1,
            "arity": [0],
            "start_nonterminal": 0,
            "productions": [{"nonterminal": 0, "symbol": 0, "children": [0]}],
        },
        {
            "nonterminal_count": 1,
            "arity": [0],
            "start_nonterminal": 0,
            "productions": [{"nonterminal": 0, "symbol": 0, "children": []}],
        },
    ],
)
def test_grammar_rejects_noncanonical_or_invalid_rules(payload: dict[str, Any]) -> None:
    if len(payload["productions"]) == 1 and not payload["productions"][0]["children"]:
        payload["productions"][0]["nonterminal"] = True
    with pytest.raises(ValidationError):
        RegularTreeGrammar.model_validate(payload)


def test_operation_is_published_with_valid_example() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "regular_tree_grammar.to_automaton.compute"
    )
    assert operation.result_type is RegularTreeGrammarToAutomatonResult
    assert len(operation.examples) == 1
