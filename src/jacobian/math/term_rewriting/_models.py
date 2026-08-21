"""Typed wire contracts for first-order term rewriting operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.term_rewriting.operations import (
    apply_substitution,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
    unify,
)
from jacobian.math.term_rewriting.values import (
    MAX_RULES,
    RankedSignature,
    RewriteApplication,
    RewriteRule,
    Substitution,
    Term,
)


class SubstitutionRequest(StrictModel):
    """Apply a substitution to a term."""

    signature: RankedSignature
    term: Term
    substitution: Substitution

    @model_validator(mode="after")
    def require_signature(self) -> Self:
        self.signature.validate_term(self.term)
        for replacement in self.substitution.mapping.values():
            self.signature.validate_term(replacement)
        return self


class SubstitutionResult(SubstitutionRequest):
    """The term after substitution."""

    result: Term

    @model_validator(mode="after")
    def bind_substitution(self) -> Self:
        self.signature.validate_term(self.result)
        if self.result != apply_substitution(self.term, self.substitution.mapping):
            raise ValueError("substitution result is not bound to its source")
        return self


class MatchingRequest(StrictModel):
    """Match a pattern against a subject term (one-way matching)."""

    signature: RankedSignature
    pattern: Term
    subject: Term

    @model_validator(mode="after")
    def require_signature(self) -> Self:
        self.signature.validate_term(self.pattern)
        self.signature.validate_term(self.subject)
        return self


class MatchingResult(MatchingRequest):
    """Result of one-way matching."""

    matched: bool
    substitution: dict[int, Term] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bind_matching(self) -> Self:
        from jacobian.math.term_rewriting.operations import match

        expected = match(self.pattern, self.subject)
        if self.matched != (expected is not None) or self.substitution != (
            expected or {}
        ):
            raise ValueError("matching result is not bound to its signed terms")
        return self


class UnificationRequest(StrictModel):
    """Unify two terms."""

    signature: RankedSignature
    left: Term
    right: Term

    @model_validator(mode="after")
    def require_signature(self) -> Self:
        self.signature.validate_term(self.left)
        self.signature.validate_term(self.right)
        return self


class UnificationResult(UnificationRequest):
    """Result of unification."""

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

    signature: RankedSignature
    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    selection: RewriteStepSelection | None = None

    @model_validator(mode="after")
    def require_valid_selection(self) -> Self:
        self.signature.validate_term(self.term)
        for rule in self.rules:
            self.signature.validate_term(rule.lhs)
            self.signature.validate_term(rule.rhs)
        if self.selection is None:
            return self
        if self.selection.rule_index >= len(self.rules):
            raise ValueError("selected rule_index is out of range")
        term_at_position(self.term, self.selection.position)
        return self


class RewriteStepResult(StrictModel):
    """All applicable derivations or the declared selected derivation."""

    signature: RankedSignature
    source_term: Term
    rules: tuple[RewriteRule, ...]
    selection: RewriteStepSelection | None
    scope: Literal["ALL_APPLICABLE_STEPS", "SELECTED_STEP"]
    applications: tuple[RewriteApplication, ...]

    @model_validator(mode="after")
    def require_exact_applications(self) -> Self:
        self.signature.validate_term(self.source_term)
        for rule in self.rules:
            self.signature.validate_term(rule.lhs)
            self.signature.validate_term(rule.rhs)
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

    signature: RankedSignature
    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    strategy: Literal["LEFTMOST_OUTERMOST_RULE_ORDER"]
    max_steps: int = Field(default=1000, ge=1, le=1000)

    @model_validator(mode="after")
    def require_signature(self) -> Self:
        self.signature.validate_term(self.term)
        for rule in self.rules:
            self.signature.validate_term(rule.lhs)
            self.signature.validate_term(rule.rhs)
        return self


class NormalFormResult(StrictModel):
    """A proved normal form or a bounded prefix with an explicit next step."""

    signature: RankedSignature
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
        self.signature.validate_term(self.source_term)
        for rule in self.rules:
            self.signature.validate_term(rule.lhs)
            self.signature.validate_term(rule.rhs)
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
