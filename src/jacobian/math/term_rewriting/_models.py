"""Typed wire contracts for first-order term rewriting operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalizationError, CanonicalLimits, canonicalize_json
from jacobian.math.term_rewriting.operations import (
    _bounded_unify,
    _validate_critical_pair_source,
    apply_substitution,
    critical_pairs,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
)
from jacobian.math.term_rewriting.values import (
    MAX_CRITICAL_PAIR_RESULT_BYTES,
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


def _require_transport_safe_depth(*terms: Term) -> None:
    """Reject term paths deeper than strict JSON transport can carry.

    The shared canonical profile caps JSON nesting at 64 levels and each
    serialized term node costs one object level plus one ``children`` array
    level inside a request, so any root-to-leaf path carries at most
    ``MAX_TERM_DEPTH`` nodes. Wire contracts enforce this iteratively so
    rejection stays typed instead of surfacing as a transport failure after
    schema admission.
    """
    stack = [(term, 1) for term in terms]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_TERM_DEPTH:
            raise _validation_error(
                "transport_depth",
                "term depth exceeds the transport-safe bound; any "
                f"root-to-leaf path carries at most {MAX_TERM_DEPTH} nodes",
            )
        stack.extend((child, depth + 1) for child in current.children)


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
        _require_transport_safe_depth(
            self.term,
            *self.substitution.mapping.values(),
            apply_substitution(self.term, self.substitution.mapping),
        )
        return self


class SubstitutionResult(SubstitutionRequest):
    """The term after substitution."""

    result: Term

    @model_validator(mode="after")
    def bind_substitution(self) -> Self:
        self.signature.validate_term(self.result)
        if self.result != apply_substitution(self.term, self.substitution.mapping):
            raise _validation_error(
                "substitution_result", "substitution result is not bound to its source"
            )
        _require_transport_safe_depth(self.result)
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
        _require_transport_safe_depth(self.pattern, self.subject)
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
            raise _validation_error(
                "matching_result", "matching result is not bound to its signed terms"
            )
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
        _require_transport_safe_depth(self.left, self.right)
        try:
            expected = _bounded_unify(self.left, self.right)
        except ValueError as error:
            raise _validation_error("unification_bound", str(error)) from error
        if expected is not None:
            _require_transport_safe_depth(*expected.values())
            try:
                canonicalize_json(
                    {
                        "signature": self.signature.model_dump(mode="json"),
                        "left": self.left.model_dump(mode="json"),
                        "right": self.right.model_dump(mode="json"),
                        "unified": True,
                        "substitution": {
                            str(variable): term.model_dump(mode="json")
                            for variable, term in expected.items()
                        },
                    },
                    limits=CanonicalLimits(
                        max_output_bytes=MAX_CRITICAL_PAIR_RESULT_BYTES
                    ),
                )
            except CanonicalizationError as error:
                raise _validation_error(
                    "unification_transport",
                    "unification result exceeds the supported transport bound",
                ) from error
        return self


class UnificationResult(UnificationRequest):
    """Result of unification."""

    unified: bool
    substitution: dict[int, Term] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_exact_unifier(self) -> Self:
        expected = _bounded_unify(self.left, self.right)
        if expected is None:
            if self.unified or self.substitution:
                raise _validation_error(
                    "failed_unification",
                    "failed unification must not claim a substitution",
                )
        elif not self.unified or self.substitution != expected:
            raise _validation_error(
                "unifier_mismatch", "substitution must be the computed idempotent MGU"
            )
        return self


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

    @model_validator(mode="after")
    def require_valid_selection(self) -> Self:
        self.signature.validate_term(self.term)
        for rule in self.rules:
            self.signature.validate_term(rule.lhs)
            self.signature.validate_term(rule.rhs)
        _require_transport_safe_depth(
            self.term, *(side for rule in self.rules for side in (rule.lhs, rule.rhs))
        )
        if self.selection is None:
            applications = rewrite_steps(self.term, self.rules)
        else:
            if self.selection.rule_index >= len(self.rules):
                raise _validation_error(
                    "selection_rule_index", "selected rule_index is out of range"
                )
            try:
                term_at_position(self.term, self.selection.position)
                application = selected_rewrite_step(
                    self.term,
                    self.rules,
                    self.selection.position,
                    self.selection.rule_index,
                )
            except ValueError as error:
                raise _validation_error("selection_position", str(error)) from error
            applications = () if application is None else (application,)
        _require_transport_safe_depth(
            *(application.term for application in applications)
        )
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
        _require_transport_safe_depth(
            self.source_term,
            *(side for rule in self.rules for side in (rule.lhs, rule.rhs)),
        )
        _require_transport_safe_depth(
            *(application.term for application in self.applications)
        )
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
            raise _validation_error(
                "scope_selection", "scope must agree with selection"
            )
        if self.applications != expected:
            raise _validation_error(
                "applications_mismatch",
                "applications do not match the declared rewrite scope",
            )
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
        _require_transport_safe_depth(
            self.term, *(side for rule in self.rules for side in (rule.lhs, rule.rhs))
        )
        normalized, _, _, next_step = normal_form(self.term, self.rules, self.max_steps)
        observed = [normalized]
        if next_step is not None:
            observed.append(next_step.term)
        _require_transport_safe_depth(*observed)
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
        _require_transport_safe_depth(
            self.source_term,
            *(side for rule in self.rules for side in (rule.lhs, rule.rhs)),
        )
        observed = [self.term]
        if self.next_step is not None:
            observed.append(self.next_step.term)
        _require_transport_safe_depth(*observed)
        term, status, steps, next_step = normal_form(
            self.source_term, self.rules, self.max_steps
        )
        if (self.term, self.status, self.steps, self.next_step) != (
            term,
            status,
            steps,
            next_step,
        ):
            raise _validation_error(
                "normal_form_replay", "normal-form result does not replay exactly"
            )
        return self


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

    @model_validator(mode="after")
    def require_bounded_critical_pair_source(self) -> Self:
        try:
            _validate_critical_pair_source(self.signature, self.rules)
        except ValueError as error:
            raise _validation_error("critical_pair_source", str(error)) from error
        _require_transport_safe_depth(
            *(side for rule in self.rules for side in (rule.lhs, rule.rhs))
        )
        return self


class CriticalPairsResult(CriticalPairsRequest):
    """The complete exact critical-pair family for a bounded finite TRS."""

    profile: CriticalPairProfile

    @model_validator(mode="after")
    def bind_critical_pairs(self) -> Self:
        expected = critical_pairs(self.signature, self.rules)
        if self.profile != expected:
            raise _validation_error(
                "critical_pairs_replay",
                "critical pairs do not replay from the source rules",
            )
        return self


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
