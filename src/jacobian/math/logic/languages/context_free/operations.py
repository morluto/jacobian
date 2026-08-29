"""Exact operations on finite context-free grammars."""

from jacobian.math.logic.languages.context_free._models import FiniteCFGO


def nullable_nonterminals(grammar: FiniteCFGO) -> tuple[bool, ...]:
    """Return nullability aligned with the grammar's nonterminal order."""

    terminals = set(grammar.terminals)
    nullable = dict.fromkeys(grammar.nonterminals, False)
    changed = True
    while changed:
        changed = False
        for rule in grammar.rules:
            if nullable[rule.head]:
                continue
            if all(
                symbol not in terminals and nullable[symbol] for symbol in rule.body
            ):
                nullable[rule.head] = True
                changed = True
    return tuple(nullable[nonterminal] for nonterminal in grammar.nonterminals)


def dependency_edges(grammar: FiniteCFGO) -> tuple[tuple[str, str], ...]:
    """Return the canonical nonterminal dependency edges."""

    nonterminals = set(grammar.nonterminals)
    return tuple(
        sorted(
            {
                (rule.head, symbol)
                for rule in grammar.rules
                for symbol in rule.body
                if symbol in nonterminals
            }
        )
    )


def first_sets(grammar: FiniteCFGO) -> tuple[tuple[str, ...], ...]:
    """Return FIRST sets aligned with the grammar's nonterminal order."""

    terminals = set(grammar.terminals)
    nonterminals = set(grammar.nonterminals)
    nullable = dict(
        zip(grammar.nonterminals, nullable_nonterminals(grammar), strict=True)
    )
    first: dict[str, set[str]] = {
        nonterminal: set() for nonterminal in grammar.nonterminals
    }
    changed = True
    while changed:
        changed = False
        for rule in grammar.rules:
            for symbol in rule.body:
                if symbol in terminals:
                    if symbol not in first[rule.head]:
                        first[rule.head].add(symbol)
                        changed = True
                    break
                if symbol not in nonterminals:
                    break
                additions = first[symbol] - first[rule.head]
                if additions:
                    first[rule.head].update(additions)
                    changed = True
                if not nullable[symbol]:
                    break
    return tuple(
        tuple(sorted(first[nonterminal])) for nonterminal in grammar.nonterminals
    )


__all__ = ["dependency_edges", "first_sets", "nullable_nonterminals"]
