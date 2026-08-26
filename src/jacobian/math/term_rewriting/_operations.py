"""Domain adapter for term rewriting operations."""

from __future__ import annotations

from typing import Literal

from jacobian.math.term_rewriting._kernel import (
    _bounded_unify,
    apply_substitution,
    critical_pairs,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
)
from jacobian.math.term_rewriting._models import (
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
)

__all__ = [
    "compute_critical_pairs",
    "compute_matching",
    "compute_normal_form",
    "compute_rewrite_step",
    "compute_substitution",
    "compute_unification",
    "verify_critical_pairs_result",
    "verify_matching_result",
    "verify_normal_form_result",
    "verify_rewrite_step_result",
    "verify_substitution_result",
    "verify_unification_result",
]


def compute_substitution(request: SubstitutionRequest) -> SubstitutionResult:
    return SubstitutionResult._from_kernel(
        request, apply_substitution(request.term, request.substitution.mapping)
    )


def compute_matching(request: MatchingRequest) -> MatchingResult:
    result = match(request.pattern, request.subject)
    if result is None:
        return MatchingResult._from_kernel(request, False, {})
    return MatchingResult._from_kernel(request, True, result)


def compute_unification(request: UnificationRequest) -> UnificationResult:
    result = _bounded_unify(request.left, request.right)
    if result is None:
        return UnificationResult._from_kernel(request, False, {})
    return UnificationResult._from_kernel(request, True, result)


def compute_rewrite_step(request: RewriteStepRequest) -> RewriteStepResult:
    scope: Literal["ALL_APPLICABLE_STEPS", "SELECTED_STEP"]
    if request.selection is None:
        applications = rewrite_steps(request.term, request.rules)
        scope = "ALL_APPLICABLE_STEPS"
    else:
        application = selected_rewrite_step(
            request.term,
            request.rules,
            request.selection.position,
            request.selection.rule_index,
        )
        applications = () if application is None else (application,)
        scope = "SELECTED_STEP"
    return RewriteStepResult._from_kernel(request, scope, applications)


def compute_normal_form(request: NormalFormRequest) -> NormalFormResult:
    term, status, steps, next_step = normal_form(
        request.term, request.rules, request.max_steps
    )
    return NormalFormResult._from_kernel(request, term, status, steps, next_step)


def compute_critical_pairs(request: CriticalPairsRequest) -> CriticalPairsResult:
    return CriticalPairsResult._from_kernel(
        request, critical_pairs(request.signature, request.rules)
    )


def verify_substitution_result(result: SubstitutionResult) -> bool:
    """Check one admitted externally supplied substitution result."""

    return result.result == apply_substitution(result.term, result.substitution.mapping)


def verify_matching_result(result: MatchingResult) -> bool:
    """Check one admitted externally supplied matching claim."""

    expected = match(result.pattern, result.subject)
    return (result.matched, result.substitution) == (
        expected is not None,
        expected or {},
    )


def verify_unification_result(result: UnificationResult) -> bool:
    """Check one admitted externally supplied MGU claim."""

    expected = _bounded_unify(result.left, result.right)
    return (result.unified, result.substitution) == (
        expected is not None,
        expected or {},
    )


def verify_rewrite_step_result(result: RewriteStepResult) -> bool:
    """Check an admitted rewrite-step result against its declared scope."""

    if result.selection is None:
        return (
            result.scope == "ALL_APPLICABLE_STEPS"
            and result.applications == rewrite_steps(result.source_term, result.rules)
        )
    application = selected_rewrite_step(
        result.source_term,
        result.rules,
        result.selection.position,
        result.selection.rule_index,
    )
    return result.scope == "SELECTED_STEP" and result.applications == (
        () if application is None else (application,)
    )


def verify_normal_form_result(result: NormalFormResult) -> bool:
    """Check one bounded normal-form trace result."""

    return (result.term, result.status, result.steps, result.next_step) == normal_form(
        result.source_term, result.rules, result.max_steps
    )


def verify_critical_pairs_result(result: CriticalPairsResult) -> bool:
    """Check one admitted complete critical-pair profile."""

    return result.profile == critical_pairs(result.signature, result.rules)
