"""Tests for context-free language operations."""

from collections.abc import Mapping

import pytest

from jacobian.math.logic.languages.context_free import (
    dependency_edges,
    first_sets,
    nullable_nonterminals,
)
from jacobian.math.logic.languages.context_free._models import (
    DependencyGraphRequest,
    FiniteCFGO,
    FirstSetsRequest,
    SymbolProfilesRequest,
)
from jacobian.math.logic.languages.context_free._tools import (
    TOOLS,
    compute_dependency_graph,
    compute_first_sets,
    compute_symbol_profiles,
)

GRAMMAR = {
    "nonterminals": ["S", "A"],
    "terminals": ["a", "b"],
    "rules": [
        {"head": "S", "body": ["A", "a"]},
        {"head": "A", "body": ["b"]},
        {"head": "A", "body": []},
    ],
    "start_symbol": "S",
}

GRAMMAR2 = {
    "nonterminals": ["S"],
    "terminals": ["a"],
    "rules": [
        {"head": "S", "body": ["a", "S"]},
        {"head": "S", "body": []},
    ],
    "start_symbol": "S",
}


def _grammar(payload: Mapping[str, object]) -> FiniteCFGO:
    return FiniteCFGO.model_validate(payload)


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "grammar.symbol_profiles.compute",
        "grammar.dependency_graph.compute",
        "grammar.first_sets.compute",
    }


def test_symbol_profiles_nullable() -> None:
    request = SymbolProfilesRequest(grammar=_grammar(GRAMMAR))
    result = compute_symbol_profiles(request)
    assert result.nullable == (False, True)
    assert nullable_nonterminals(request.grammar) == result.nullable


def test_symbol_profiles_nullable_simple() -> None:
    request = SymbolProfilesRequest(grammar=_grammar(GRAMMAR2))
    result = compute_symbol_profiles(request)
    assert result.nullable == (True,)


def test_dependency_graph() -> None:
    request = DependencyGraphRequest(grammar=_grammar(GRAMMAR))
    result = compute_dependency_graph(request)
    assert ("S", "A") in result.edges
    assert dependency_edges(request.grammar) == result.edges


def test_first_sets() -> None:
    request = FirstSetsRequest(grammar=_grammar(GRAMMAR2))
    result = compute_first_sets(request)
    assert result.first_sets == (("a",),)
    assert first_sets(request.grammar) == result.first_sets


def test_first_sets_nullable_prefix() -> None:
    """FIRST(S) must include symbols after a nullable nonterminal prefix.

    For S -> A a, A -> b | epsilon, FIRST(S) = {a, b} because A is nullable
    so the terminal a following A is also reachable.
    """
    request = FirstSetsRequest(grammar=_grammar(GRAMMAR))
    result = compute_first_sets(request)
    assert result.first_sets == (("a", "b"), ("b",))


def test_first_sets_nullable_prefix_two_symbol() -> None:
    """FIRST propagates through a chain of nullable nonterminals.

    S -> A B c, A -> epsilon, B -> epsilon yields FIRST(S) = {c} because
    both A and B are nullable so c is reachable.
    """
    grammar = {
        "nonterminals": ["S", "A", "B"],
        "terminals": ["c"],
        "rules": [
            {"head": "S", "body": ["A", "B", "c"]},
            {"head": "A", "body": []},
            {"head": "B", "body": []},
        ],
        "start_symbol": "S",
    }
    request = FirstSetsRequest(grammar=_grammar(grammar))
    result = compute_first_sets(request)
    assert result.first_sets == (("c",), (), ())


def test_first_sets_nullable_prefix_mixed() -> None:
    """FIRST stops at the first non-nullable nonterminal.

    S -> A B, A -> a | epsilon, B -> b gives FIRST(S) = {a, b}: A is
    nullable so b from B is reachable, but B is not nullable so nothing
    further is added.
    """
    grammar = {
        "nonterminals": ["S", "A", "B"],
        "terminals": ["a", "b"],
        "rules": [
            {"head": "S", "body": ["A", "B"]},
            {"head": "A", "body": ["a"]},
            {"head": "A", "body": []},
            {"head": "B", "body": ["b"]},
        ],
        "start_symbol": "S",
    }
    request = FirstSetsRequest(grammar=_grammar(grammar))
    result = compute_first_sets(request)
    assert result.first_sets == (("a", "b"), ("a",), ("b",))


def test_grammar_rejects_undeclared_body_symbol() -> None:
    """A rule body referencing an undeclared symbol is a validation error."""
    grammar = {
        "nonterminals": ["S"],
        "terminals": ["a"],
        "rules": [{"head": "S", "body": ["X"]}],
        "start_symbol": "S",
    }
    with pytest.raises(ValueError):
        FiniteCFGO.model_validate(grammar)


def test_grammar_rejects_overlapping_terminal_nonterminal() -> None:
    """A symbol that is both a terminal and a nonterminal is rejected."""
    grammar = {
        "nonterminals": ["S"],
        "terminals": ["S"],
        "rules": [{"head": "S", "body": ["S"]}],
        "start_symbol": "S",
    }
    with pytest.raises(ValueError):
        FiniteCFGO.model_validate(grammar)


def test_grammar_accepts_declared_symbols() -> None:
    """A grammar where every body symbol is declared is accepted."""
    grammar = {
        "nonterminals": ["S", "A"],
        "terminals": ["a", "b"],
        "rules": [
            {"head": "S", "body": ["A", "a"]},
            {"head": "A", "body": ["b"]},
            {"head": "A", "body": []},
        ],
        "start_symbol": "S",
    }
    FiniteCFGO.model_validate(grammar)
