"""Typed wire contracts for first-order term rewriting operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.term_rewriting.values import (
    MAX_CRITICAL_PAIR_RULES,
    MAX_RULES,
    MAX_TERM_DEPTH,
    MAX_VARIABLE_LABEL,
    CriticalPairProfile,
    RankedSignature,
    RewriteApplication,
    RewriteRule,
    Substitution,
    Term,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by term-rewriting contracts."""

    return PydanticCustomError(f"term_rewriting.{reason}", message)


class SubstitutionRequest(StrictModel):
    """Apply a substitution to a term."""

    signature: RankedSignature
    term: Term
    substitution: Substitution


class SubstitutionResult(SubstitutionRequest):
    """The term after substitution."""

    result: Term

    @classmethod
    def _from_kernel(
        cls,
        *,
        signature: RankedSignature,
        term: Term,
        substitution: Substitution,
        result: Term,
    ) -> Self:
        return cls.model_construct(
            signature=signature,
            term=term,
            substitution=substitution,
            result=result,
        )


class MatchingRequest(StrictModel):
    """Match a pattern against a subject term (one-way matching)."""

    signature: RankedSignature
    pattern: Term
    subject: Term


class MatchingResult(MatchingRequest):
    """Result of one-way matching."""

    matched: bool
    substitution: dict[int, Term] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bind_matching(self) -> Self:
        if not self.matched and self.substitution:
            raise _validation_error(
                "matching_result", "an unmatched result cannot include a substitution"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        signature: RankedSignature,
        pattern: Term,
        subject: Term,
        matched: bool,
        substitution: dict[int, Term],
    ) -> Self:
        return cls.model_construct(
            signature=signature,
            pattern=pattern,
            subject=subject,
            matched=matched,
            substitution=substitution,
        )


class UnificationRequest(StrictModel):
    """Unify two terms."""

    signature: RankedSignature
    left: Term
    right: Term


class UnificationResult(UnificationRequest):
    """Result of unification."""

    unified: bool
    substitution: dict[int, Term] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_exact_unifier(self) -> Self:
        if not self.unified and self.substitution:
            raise _validation_error(
                "failed_unification", "failed unification must not claim a substitution"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        signature: RankedSignature,
        left: Term,
        right: Term,
        unified: bool,
        substitution: dict[int, Term],
    ) -> Self:
        return cls.model_construct(
            signature=signature,
            left=left,
            right=right,
            unified=unified,
            substitution=substitution,
        )


class RewriteStepSelection(StrictModel):
    """An agent-selected redex and rule for one rewrite derivation."""

    position: tuple[int, ...]
    rule_index: int = Field(ge=0)

    @model_validator(mode="after")
    def require_nonnegative_position(self) -> Self:
        if any(child_index < 0 for child_index in self.position):
            raise _validation_error(
                "negative_position", "rewrite position indices must be non-negative"
            )
        return self


class RewriteStepRequest(StrictModel):
    """Enumerate every step, or apply one explicitly selected redex and rule."""

    signature: RankedSignature
    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    selection: RewriteStepSelection | None = None


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
        expected_scope = (
            "ALL_APPLICABLE_STEPS" if self.selection is None else "SELECTED_STEP"
        )
        if self.scope != expected_scope:
            raise _validation_error(
                "scope_selection", "scope must agree with selection"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        signature: RankedSignature,
        source_term: Term,
        rules: tuple[RewriteRule, ...],
        selection: RewriteStepSelection | None,
        scope: Literal["ALL_APPLICABLE_STEPS", "SELECTED_STEP"],
        applications: tuple[RewriteApplication, ...],
    ) -> Self:
        return cls.model_construct(
            signature=signature,
            source_term=source_term,
            rules=rules,
            selection=selection,
            scope=scope,
            applications=applications,
        )


class NormalFormRequest(StrictModel):
    """Run an explicit bounded normalization strategy."""

    signature: RankedSignature
    term: Term
    rules: tuple[RewriteRule, ...] = Field(min_length=1, max_length=MAX_RULES)
    strategy: Literal["LEFTMOST_OUTERMOST_RULE_ORDER"]
    max_steps: int = Field(default=1000, ge=1, le=1000)


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
        if self.status == "NORMAL_FORM" and self.next_step is not None:
            raise _validation_error(
                "normal_form_shape", "a normal form cannot carry a next step"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        signature: RankedSignature,
        source_term: Term,
        rules: tuple[RewriteRule, ...],
        strategy: Literal["LEFTMOST_OUTERMOST_RULE_ORDER"],
        max_steps: int,
        term: Term,
        status: Literal["NORMAL_FORM", "STEP_LIMIT"],
        steps: int,
        next_step: RewriteApplication | None,
    ) -> Self:
        return cls.model_construct(
            signature=signature,
            source_term=source_term,
            rules=rules,
            strategy=strategy,
            max_steps=max_steps,
            term=term,
            status=status,
            steps=steps,
            next_step=next_step,
        )


class CriticalPairsRequest(StrictModel):
    """Enumerate every nontrivial source-indexed critical pair of a finite TRS."""

    signature: RankedSignature = Field(
        description=(
            "Finite ranked signature for every rule; function symbols must use "
            "these arities exactly."
        )
    )
    rules: tuple[RewriteRule, ...] = Field(
        max_length=MAX_CRITICAL_PAIR_RULES,
        description=(
            "Duplicate-free ordered rules, possibly empty. Variable labels "
            "are non-negative integers up to the interoperable JSON integer "
            "maximum published as the term symbol maximum; their serialized "
            "width is charged against the byte bound. Terms carry at most "
            "31 nodes on any root-to-leaf path, the deepest chain strict "
            "JSON transport accepts. The complete ordered nonvariable "
            "overlap ledger has at most 32 candidates. The materialized "
            "replay work, the retained result, and its exact replay are "
            "bounded by 42752 structural nodes and 4MiB. Composed reducts "
            "and pair substitution bindings carry at most 30 nodes on any "
            "root-to-leaf path; deeper compositions reject at admission."
        ),
        json_schema_extra={
            "x-jacobian-bounds": {
                "max_overlap_candidates": 32,
                "max_result_nodes": 42752,
                "max_result_bytes": 4194304,
                "max_variable_label": MAX_VARIABLE_LABEL,
                "max_term_depth": MAX_TERM_DEPTH,
                "max_result_term_depth": MAX_TERM_DEPTH - 1,
            }
        },
    )


class CriticalPairsResult(CriticalPairsRequest):
    """The complete exact critical-pair family for a bounded finite TRS."""

    profile: CriticalPairProfile

    @classmethod
    def _from_kernel(
        cls,
        *,
        signature: RankedSignature,
        rules: tuple[RewriteRule, ...],
        profile: CriticalPairProfile,
    ) -> Self:
        return cls.model_construct(signature=signature, rules=rules, profile=profile)


__all__ = [
    "CriticalPairsRequest",
    "CriticalPairsResult",
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
