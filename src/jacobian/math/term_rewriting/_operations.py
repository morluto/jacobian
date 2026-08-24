"""Domain adapter for term rewriting operations."""

from __future__ import annotations

from typing import Literal

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
from jacobian.math.term_rewriting.operations import (
    _bounded_unify,
    apply_substitution,
    critical_pairs,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
)

__all__ = [
    "compute_critical_pairs",
    "compute_matching",
    "compute_normal_form",
    "compute_rewrite_step",
    "compute_substitution",
    "compute_unification",
]


def compute_substitution(request: SubstitutionRequest) -> SubstitutionResult:
    return SubstitutionResult(
        **request.model_dump(),
        result=apply_substitution(request.term, request.substitution.mapping),
    )


def compute_matching(request: MatchingRequest) -> MatchingResult:
    result = match(request.pattern, request.subject)
    if result is None:
        return MatchingResult(**request.model_dump(), matched=False, substitution={})
    return MatchingResult(**request.model_dump(), matched=True, substitution=result)


def compute_unification(request: UnificationRequest) -> UnificationResult:
    result = _bounded_unify(request.left, request.right)
    if result is None:
        return UnificationResult(
            signature=request.signature,
            left=request.left,
            right=request.right,
            unified=False,
            substitution={},
        )
    return UnificationResult(
        signature=request.signature,
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
        signature=request.signature,
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
        signature=request.signature,
        source_term=request.term,
        rules=request.rules,
        strategy=request.strategy,
        max_steps=request.max_steps,
        term=term,
        status=status,
        steps=steps,
        next_step=next_step,
    )


def compute_critical_pairs(request: CriticalPairsRequest) -> CriticalPairsResult:
    return CriticalPairsResult(
        signature=request.signature,
        rules=request.rules,
        profile=critical_pairs(request.signature, request.rules),
    )
