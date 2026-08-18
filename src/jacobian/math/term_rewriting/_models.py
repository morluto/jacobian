"""Typed wire contracts for first-order term rewriting operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.term_rewriting.operations import (
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
    unify,
)
from jacobian.math.term_rewriting.values import (
    MAX_RULES,
    RewriteApplication,
    RewriteRule,
    Substitution,
    Term,
)


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

    left: Term
    right: Term
    unified: bool
    substitution: dict[int, Term] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_exact_unifier(self) -> Self:
        expected = unify(self.left, self.right)
        if expected is None:
            if self.unified or self.substitution:
                raise ValueError("failed unification must not claim a substitution")
        elif not self.unified or self.substitution != expected:
            raise ValueError("substitution must be the computed idempotent MGU")
        return self


class RewriteStepSelection(StrictModel):
    """An agent-selected redex and rule for one rewrite derivation."""

    position: tuple[int, ...]
    rule_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_nonnegative_position(self) -> Self:
        if any(child_index < 0 for child_index in self.position):
            raise ValueError("rewrite position indices must be non-negative")
        return self


class RewriteStepRequest(StrictModel):
    """Enumerate every step, or apply one explicitly selected redex and rule."""

    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    selection: RewriteStepSelection | None = None

    @model_validator(mode="after")
    def require_valid_selection(self) -> Self:
        if self.selection is None:
            return self
        if self.selection.rule_index >= len(self.rules):
            raise ValueError("selected rule_index is out of range")
        term_at_position(self.term, self.selection.position)
        return self


class RewriteStepResult(StrictModel):
    """All applicable derivations or the declared selected derivation."""

    source_term: Term
    rules: tuple[RewriteRule, ...]
    selection: RewriteStepSelection | None
    scope: Literal["ALL_APPLICABLE_STEPS", "SELECTED_STEP"]
    applications: tuple[RewriteApplication, ...]

    @model_validator(mode="after")
    def require_exact_applications(self) -> Self:
        if self.selection is None:
            expected = rewrite_steps(self.source_term, self.rules)
            expected_scope = "ALL_APPLICABLE_STEPS"
        else:
            application = selected_rewrite_step(
                self.source_term,
                self.rules,
                self.selection.position,
                self.selection.rule_index,
            )
            expected = () if application is None else (application,)
            expected_scope = "SELECTED_STEP"
        if self.scope != expected_scope:
            raise ValueError("scope must agree with selection")
        if self.applications != expected:
            raise ValueError("applications do not match the declared rewrite scope")
        return self


class NormalFormRequest(StrictModel):
    """Run an explicit bounded normalization strategy."""

    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    strategy: Literal["LEFTMOST_OUTERMOST_RULE_ORDER"]
    max_steps: int = Field(default=1000, ge=1, le=1000)


class NormalFormResult(StrictModel):
    """A proved normal form or a bounded prefix with an explicit next step."""

    source_term: Term
    rules: tuple[RewriteRule, ...]
    strategy: Literal["LEFTMOST_OUTERMOST_RULE_ORDER"]
    max_steps: int = Field(ge=1, le=1000)
    term: Term
    status: Literal["NORMAL_FORM", "STEP_LIMIT"]
    steps: int = Field(ge=0)
    next_step: RewriteApplication | None

    @model_validator(mode="after")
    def require_exact_bounded_run(self) -> Self:
        term, status, steps, next_step = normal_form(
            self.source_term, self.rules, self.max_steps
        )
        if (self.term, self.status, self.steps, self.next_step) != (
            term,
            status,
            steps,
            next_step,
        ):
            raise ValueError("normal-form result does not replay exactly")
        return self


__all__ = [
    "MatchingRequest",
    "MatchingResult",
    "NormalFormRequest",
    "NormalFormResult",
    "RewriteStepRequest",
    "RewriteStepResult",
    "RewriteStepSelection",
    "SubstitutionRequest",
    "SubstitutionResult",
    "UnificationRequest",
    "UnificationResult",
]
