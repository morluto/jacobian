"""Provider-independent values for first-order term rewriting."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_TERMS = 32
MAX_SYMBOLS = 64
MAX_ARITY = 16
MAX_RULES = 64


def _variable_symbols(term: Term) -> set[int]:
    if term.is_variable:
        return {term.symbol}
    return set().union(*(_variable_symbols(child) for child in term.children))


class Term(StrictModel):
    """A first-order term represented as a tree.

    A term is either a variable (``is_variable=True``) or a function
    application (``symbol`` applied to ``children``, which are themselves
    terms).
    """

    is_variable: bool = False
    symbol: int = Field(ge=0)
    children: tuple[Term, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_term(self) -> Self:
        if self.is_variable and self.children:
            raise ValueError("a variable cannot have children")
        if not self.is_variable and self.symbol < 0:
            raise ValueError("function symbol must be non-negative")
        if len(self.children) > MAX_ARITY:
            raise ValueError("too many children (arity exceeds bound)")
        return self


class RewriteRule(StrictModel):
    """A rewrite rule: left-hand side rewrites to right-hand side."""

    lhs: Term
    rhs: Term

    @model_validator(mode="after")
    def require_valid_rule(self) -> Self:
        if self.lhs.is_variable:
            raise ValueError("LHS must be a function application, not a variable")
        extra_variables = _variable_symbols(self.rhs) - _variable_symbols(self.lhs)
        if extra_variables:
            raise ValueError("RHS variables must occur in the LHS")
        return self


class RewriteApplication(StrictModel):
    """One fully witnessed one-step rewrite derivation."""

    position: tuple[int, ...]
    rule_index: int = Field(ge=0)
    substitution: dict[int, Term]
    term: Term


Term.model_rebuild()


class Substitution(StrictModel):
    """A variable substitution mapping variable IDs to terms."""

    mapping: dict[int, Term] = Field(default_factory=dict)


__all__ = [
    "MAX_ARITY",
    "MAX_RULES",
    "MAX_SYMBOLS",
    "MAX_TERMS",
    "RewriteApplication",
    "RewriteRule",
    "Substitution",
    "Term",
]
