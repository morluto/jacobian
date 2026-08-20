"""Typed wire contracts for context-free language operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_NONTERMINALS = 32
MAX_RULES = 256


class GrammarRule(StrictModel):
    """One production rule A -> alpha."""

    head: str = Field(min_length=1, max_length=64)
    body: tuple[str, ...] = Field(min_length=0, max_length=32)


class FiniteCFGO(StrictModel):
    """A finite context-free grammar."""

    nonterminals: tuple[str, ...] = Field(min_length=1, max_length=MAX_NONTERMINALS)
    terminals: tuple[str, ...] = Field(min_length=0, max_length=MAX_NONTERMINALS)
    rules: tuple[GrammarRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    start_symbol: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.start_symbol not in self.nonterminals:
            raise ValueError("start_symbol must be a nonterminal")
        nonterminal_set = set(self.nonterminals)
        terminal_set = set(self.terminals)
        disjoint = nonterminal_set & terminal_set
        if disjoint:
            raise ValueError(
                f"terminals and nonterminals must be disjoint: {sorted(disjoint)}"
            )
        declared = nonterminal_set | terminal_set
        for rule in self.rules:
            if rule.head not in nonterminal_set:
                raise ValueError("rule heads must be nonterminals")
            for symbol in rule.body:
                if symbol not in declared:
                    raise ValueError(
                        f"rule body symbol {symbol!r} is not a declared terminal "
                        "or nonterminal"
                    )
        return self


class SymbolProfilesRequest(StrictModel):
    grammar: FiniteCFGO


class DependencyGraphRequest(StrictModel):
    grammar: FiniteCFGO


class FirstSetsRequest(StrictModel):
    grammar: FiniteCFGO


# Results


class SymbolProfilesResult(StrictModel):
    nullable: tuple[bool, ...]
    method: str = "FIXED_POINT_ITERATION"


class DependencyGraphResult(StrictModel):
    edges: tuple[tuple[str, str], ...]
    method: str = "RULE_BODY_DEPENDENCY"


class FirstSetsResult(StrictModel):
    first_sets: tuple[tuple[str, ...], ...]
    method: str = "FIXED_POINT_ITERATION"
