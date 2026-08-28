"""Domain adapter for term rewriting operations."""

from __future__ import annotations

from typing import Literal

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.term_rewriting._kernel import (
    _bounded_unify,
    _validate_critical_pair_source,
    apply_substitution,
    critical_pairs,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
)
from jacobian.math.logic.term_rewriting._models import (
    CriticalPairsRequest,
    CriticalPairsResult,
    MatchingRequest,
    MatchingResult,
    NormalFormRequest,
    NormalFormResult,
    RewriteStepRequest,
    RewriteStepResult,
    SubstitutionRequest,
    SubstitutionResult,
    UnificationRequest,
    UnificationResult,
    _require_transport_safe_depth,
)
from jacobian.math.logic.term_rewriting.values import RankedSignature, RewriteRule, Term

__all__ = [
    "compute_critical_pairs",
    "compute_matching",
    "compute_normal_form",
    "compute_rewrite_step",
    "compute_substitution",
    "compute_unification",
]


def _domain_error(
    error: ValueError,
    *,
    fallback_code: str,
    location: tuple[str | int, ...],
) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=location,
        code=getattr(error, "type", f"term_rewriting.{fallback_code}"),
        message=str(error),
    )


def _admit_terms(
    signature: RankedSignature,
    terms: tuple[Term, ...],
    *,
    location: tuple[str | int, ...],
) -> None:
    try:
        for term in terms:
            signature.validate_term(term)
        _require_transport_safe_depth(*terms)
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="signature", location=location
        ) from error


def _rule_terms(rules: tuple[RewriteRule, ...]) -> tuple[Term, ...]:
    return tuple(side for rule in rules for side in (rule.lhs, rule.rhs))


def compute_substitution(request: SubstitutionRequest) -> SubstitutionResult:
    replacements = tuple(request.substitution.mapping.values())
    _admit_terms(request.signature, (request.term, *replacements), location=("term",))
    result = apply_substitution(request.term, request.substitution.mapping)
    try:
        _require_transport_safe_depth(result)
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="transport_depth", location=("substitution",)
        ) from error
    return SubstitutionResult._from_kernel(request, result)


def compute_matching(request: MatchingRequest) -> MatchingResult:
    _admit_terms(
        request.signature, (request.pattern, request.subject), location=("pattern",)
    )
    result = match(request.pattern, request.subject)
    if result is None:
        return MatchingResult._from_kernel(request, False, {})
    return MatchingResult._from_kernel(request, True, result)


def compute_unification(request: UnificationRequest) -> UnificationResult:
    _admit_terms(request.signature, (request.left, request.right), location=("left",))
    try:
        result = _bounded_unify(request.left, request.right)
        if result is not None:
            _require_transport_safe_depth(*result.values())
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="unification_bound", location=("left", "right")
        ) from error
    if result is None:
        return UnificationResult._from_kernel(request, False, {})
    return UnificationResult._from_kernel(request, True, result)


def compute_rewrite_step(request: RewriteStepRequest) -> RewriteStepResult:
    _admit_terms(
        request.signature,
        (request.term, *_rule_terms(request.rules)),
        location=("term", "rules"),
    )
    scope: Literal["ALL_APPLICABLE_STEPS", "SELECTED_STEP"]
    try:
        if request.selection is None:
            applications = rewrite_steps(request.term, request.rules)
            scope = "ALL_APPLICABLE_STEPS"
        else:
            if request.selection.rule_index >= len(request.rules):
                raise ValueError("selected rule_index is out of range")
            term_at_position(request.term, request.selection.position)
            application = selected_rewrite_step(
                request.term,
                request.rules,
                request.selection.position,
                request.selection.rule_index,
            )
            applications = () if application is None else (application,)
            scope = "SELECTED_STEP"
        _require_transport_safe_depth(
            *(application.term for application in applications)
        )
    except ValueError as error:
        code = (
            "selection_rule_index"
            if "rule_index" in str(error)
            else "selection_position"
        )
        raise _domain_error(
            error, fallback_code=code, location=("selection",)
        ) from error
    return RewriteStepResult._from_kernel(request, scope, applications)


def compute_normal_form(request: NormalFormRequest) -> NormalFormResult:
    _admit_terms(
        request.signature,
        (request.term, *_rule_terms(request.rules)),
        location=("term", "rules"),
    )
    try:
        term, status, steps, next_step = normal_form(
            request.term, request.rules, request.max_steps
        )
        observed = (term,) if next_step is None else (term, next_step.term)
        _require_transport_safe_depth(*observed)
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="normal_form_bound", location=("term", "rules")
        ) from error
    return NormalFormResult._from_kernel(request, term, status, steps, next_step)


def compute_critical_pairs(request: CriticalPairsRequest) -> CriticalPairsResult:
    try:
        _validate_critical_pair_source(request.signature, request.rules)
        _require_transport_safe_depth(*_rule_terms(request.rules))
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="critical_pair_source", location=("rules",)
        ) from error
    return CriticalPairsResult._from_kernel(
        request, critical_pairs(request.signature, request.rules)
    )
