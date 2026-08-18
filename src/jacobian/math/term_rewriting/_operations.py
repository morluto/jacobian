"""Domain adapter for term rewriting operations."""

from __future__ import annotations

from typing import Literal

from jacobian.math.term_rewriting._models import (
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
from jacobian.math.term_rewriting.operations import (
    apply_substitution,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    unify,
)

__all__ = [
    "compute_matching",
    "compute_normal_form",
    "compute_rewrite_step",
    "compute_substitution",
    "compute_unification",
]


def compute_substitution(request: SubstitutionRequest) -> SubstitutionResult:
    return SubstitutionResult(
        term=apply_substitution(request.term, request.substitution.mapping)
    )


def compute_matching(request: MatchingRequest) -> MatchingResult:
    result = match(request.pattern, request.subject)
    if result is None:
        return MatchingResult(matched=False, substitution={})
    return MatchingResult(matched=True, substitution=result)


def compute_unification(request: UnificationRequest) -> UnificationResult:
    result = unify(request.left, request.right)
    if result is None:
        return UnificationResult(
            left=request.left,
            right=request.right,
            unified=False,
            substitution={},
        )
    return UnificationResult(
        left=request.left,
        right=request.right,
        unified=True,
        substitution=result,
    )


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
    return RewriteStepResult(
        source_term=request.term,
        rules=request.rules,
        selection=request.selection,
        scope=scope,
        applications=applications,
    )


def compute_normal_form(request: NormalFormRequest) -> NormalFormResult:
    term, status, steps, next_step = normal_form(
        request.term, request.rules, request.max_steps
    )
    return NormalFormResult(
        source_term=request.term,
        rules=request.rules,
        strategy=request.strategy,
        max_steps=request.max_steps,
        term=term,
        status=status,
        steps=steps,
        next_step=next_step,
    )
