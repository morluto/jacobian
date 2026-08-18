"""Typed wire contracts for first-order term rewriting operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.term_rewriting.values import RewriteRule, Substitution, Term


class SubstitutionRequest(StrictModel):
    """Apply a substitution to a term."""

    term: Term
    substitution: Substitution


class SubstitutionResult(StrictModel):
    """The term after substitution."""

    term: Term


class MatchingRequest(StrictModel):
    """Match a pattern against a subject term (one-way matching)."""

    pattern: Term
    subject: Term


class MatchingResult(StrictModel):
    """Result of one-way matching."""

    matched: bool
    substitution: dict[int, Term] = Field(default_factory=dict)


class UnificationRequest(StrictModel):
    """Unify two terms."""

    left: Term
    right: Term


class UnificationResult(StrictModel):
    """Result of unification."""

    unified: bool
    substitution: dict[int, Term] = Field(default_factory=dict)


class RewriteStepRequest(StrictModel):
    """Apply one rewrite step to a term."""

    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1)


class RewriteStepResult(StrictModel):
    """Result of one rewrite step."""

    rewritten: bool
    term: Term


class NormalFormRequest(StrictModel):
    """Compute the normal form of a term under a set of rules."""

    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1)
    max_steps: int = Field(default=1000, ge=1, le=100000)


class NormalFormResult(StrictModel):
    """The normal form (or last term if max_steps reached)."""

    term: Term
    converged: bool
    steps: int


__all__ = [
    "MatchingRequest",
    "MatchingResult",
    "NormalFormRequest",
    "NormalFormResult",
    "RewriteStepRequest",
    "RewriteStepResult",
    "SubstitutionRequest",
    "SubstitutionResult",
    "UnificationRequest",
    "UnificationResult",
]
