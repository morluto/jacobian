"""Provider-independent values for first-order term rewriting."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_TERMS = 32
MAX_SYMBOLS = 64
MAX_ARITY = 16
MAX_RULES = 64
MAX_VARIABLE_LABEL = (1 << 53) - 1
MAX_TERM_DEPTH = 31
MAX_CRITICAL_PAIR_RULES = 8
MAX_CRITICAL_PAIR_CANDIDATES = 32
MAX_CRITICAL_PAIR_RESULT_NODES = 42_752
MAX_CRITICAL_PAIR_RESULT_BYTES = 4 * 1024 * 1024


class RankedSignature(StrictModel):
    """A finite ordered family of function-symbol arities."""

    arities: tuple[int, ...] = Field(min_length=1, max_length=MAX_SYMBOLS)

    @model_validator(mode="after")
    def require_bounded_arities(self) -> Self:
        if any(not 0 <= arity <= MAX_ARITY for arity in self.arities):
            raise ValueError("signature arities must be within the supported bound")
        return self

    def validate_term(self, term: Term) -> None:
        stack = [term]
        while stack:
            current = stack.pop()
            if not current.is_variable:
                if current.symbol >= len(self.arities):
                    raise ValueError("term uses an undeclared function symbol")
                if len(current.children) != self.arities[current.symbol]:
                    raise ValueError("term child count must match the ranked signature")
            stack.extend(current.children)


def _variable_symbols(term: Term) -> set[int]:
    symbols: set[int] = set()
    stack = [term]
    while stack:
        current = stack.pop()
        if current.is_variable:
            symbols.add(current.symbol)
        else:
            stack.extend(current.children)
    return symbols


class Term(StrictModel):
    """A first-order term represented as a tree.

    A term is either a variable (``is_variable=True``) or a function
    application (``symbol`` applied to ``children``, which are themselves
    terms). Variable labels share ``symbol`` and are bounded by the
    interoperable JSON integer range, so every admitted term carries through
    strict JSON transport; wire contracts additionally bound every
    root-to-leaf path to ``MAX_TERM_DEPTH`` nodes, the deepest serialized
    chain that transport accepts.
    """

    is_variable: bool = False
    symbol: int = Field(ge=0, le=MAX_VARIABLE_LABEL)
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


class CriticalPair(StrictModel):
    """One source-indexed first-order critical pair.

    The two reducts are formed after the deterministic standardize-apart
    convention used by :func:`critical_pairs`: variables of the outer rule
    receive consecutive IDs in left-hand-side preorder, followed by variables
    of the inner rule.  ``inner_reduct`` rewrites at ``position`` first and
    ``outer_reduct`` rewrites the outer root first.
    """

    candidate_index: int = Field(ge=0)
    outer_variable_renaming: dict[int, int]
    inner_variable_renaming: dict[int, int]
    substitution: dict[int, Term]
    inner_reduct: Term
    outer_reduct: Term


class CriticalOverlapCandidate(StrictModel):
    """One checked source overlap, including an exact unifiability outcome."""

    outer_rule_index: int = Field(ge=0)
    inner_rule_index: int = Field(ge=0)
    position: tuple[int, ...]
    outer_variable_renaming: dict[int, int]
    inner_variable_renaming: dict[int, int]
    unifiable: bool


class CriticalPairProfile(StrictModel):
    """Complete source-indexed overlap ledger and its unifiable critical pairs."""

    candidates: tuple[CriticalOverlapCandidate, ...]
    pairs: tuple[CriticalPair, ...]


Term.model_rebuild()


class Substitution(StrictModel):
    """A variable substitution mapping variable IDs to terms."""

    mapping: dict[int, Term] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_bounded_labels(self) -> Self:
        if any(not 0 <= key <= MAX_VARIABLE_LABEL for key in self.mapping):
            raise ValueError(
                "substitution variable labels must be within the supported bound"
            )
        return self


__all__ = [
    "MAX_ARITY",
    "MAX_CRITICAL_PAIR_CANDIDATES",
    "MAX_CRITICAL_PAIR_RESULT_BYTES",
    "MAX_CRITICAL_PAIR_RESULT_NODES",
    "MAX_CRITICAL_PAIR_RULES",
    "MAX_SYMBOLS",
    "MAX_TERMS",
    "MAX_TERM_DEPTH",
    "MAX_VARIABLE_LABEL",
    "CriticalOverlapCandidate",
    "CriticalPair",
    "CriticalPairProfile",
    "RankedSignature",
    "RewriteApplication",
    "RewriteRule",
    "Substitution",
    "Term",
]
