"""Tests for context-free language operations."""

from jacobian.math.context_free_languages_ops._models import (
    DependencyGraphRequest,
    FirstSetsRequest,
    SymbolProfilesRequest,
)
from jacobian.math.context_free_languages_ops._operations import (
    compute_dependency_graph,
    compute_first_sets,
    compute_symbol_profiles,
)
from jacobian.math.context_free_languages_ops._tools import TOOLS

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


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "grammar.symbol_profiles.compute",
        "grammar.dependency_graph.compute",
        "grammar.first_sets.compute",
    }


def test_symbol_profiles_nullable() -> None:
    request = SymbolProfilesRequest(grammar=GRAMMAR)
    result = compute_symbol_profiles(request)
    assert result.nullable == (False, True)


def test_symbol_profiles_nullable_simple() -> None:
    request = SymbolProfilesRequest(grammar=GRAMMAR2)
    result = compute_symbol_profiles(request)
    assert result.nullable == (True,)


def test_dependency_graph() -> None:
    request = DependencyGraphRequest(grammar=GRAMMAR)
    result = compute_dependency_graph(request)
    assert ("S", "A") in result.edges


def test_first_sets() -> None:
    request = FirstSetsRequest(grammar=GRAMMAR2)
    result = compute_first_sets(request)
    assert result.first_sets == (("a",),)
