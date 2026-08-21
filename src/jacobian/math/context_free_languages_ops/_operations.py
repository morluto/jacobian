"""Domain functions for context-free language operations."""

from __future__ import annotations

from jacobian.math.context_free_languages_ops._models import (
    DependencyGraphRequest,
    DependencyGraphResult,
    FiniteCFGO,
    FirstSetsRequest,
    FirstSetsResult,
    SymbolProfilesRequest,
    SymbolProfilesResult,
)


def _compute_nullable(grammar: FiniteCFGO, terminal_set: set[str]) -> dict[str, bool]:
    """Return a map from nonterminal to its nullability via fixed-point iteration."""
    nullable = dict.fromkeys(grammar.nonterminals, False)
    changed = True
    while changed:
        changed = False
        for rule in grammar.rules:
            if nullable[rule.head]:
                continue
            all_nullable = True
            for symbol in rule.body:
                if symbol in terminal_set:
                    all_nullable = False
                    break
                if symbol in nullable and not nullable[symbol]:
                    all_nullable = False
                    break
            if all_nullable:
                nullable[rule.head] = True
                changed = True
    return nullable


def compute_symbol_profiles(request: SymbolProfilesRequest) -> SymbolProfilesResult:
    """Compute nullable nonterminals via fixed-point iteration."""
    grammar = request.grammar
    terminal_set = set(grammar.terminals)
    nullable = _compute_nullable(grammar, terminal_set)
    return SymbolProfilesResult(
        nullable=tuple(nullable[nt] for nt in grammar.nonterminals)
    )


def compute_dependency_graph(request: DependencyGraphRequest) -> DependencyGraphResult:
    """Compute the dependency graph: A -> B if A has a rule containing B."""
    grammar = request.grammar
    edges: set[tuple[str, str]] = set()
    for rule in grammar.rules:
        for symbol in rule.body:
            if symbol in grammar.nonterminals:
                edges.add((rule.head, symbol))
    return DependencyGraphResult(edges=tuple(sorted(edges)))


def compute_first_sets(request: FirstSetsRequest) -> FirstSetsResult:
    """Compute FIRST sets via fixed-point iteration."""
    grammar = request.grammar
    terminals = set(grammar.terminals)
    nonterminals = set(grammar.nonterminals)
    nullable = _compute_nullable(grammar, terminals)
    first: dict[str, set[str]] = {nt: set() for nt in grammar.nonterminals}
    for _ in range(256):
        changed = False
        for rule in grammar.rules:
            head = rule.head
            for symbol in rule.body:
                if symbol in terminals:
                    if symbol not in first[head]:
                        first[head].add(symbol)
                        changed = True
                    break
                elif symbol in nonterminals:
                    new = first[symbol] - first[head]
                    if new:
                        first[head] |= new
                        changed = True
                    if not nullable[symbol]:
                        break
                else:
                    break
        if not changed:
            break
    return FirstSetsResult(
        first_sets=tuple(tuple(sorted(first[nt])) for nt in grammar.nonterminals)
    )
