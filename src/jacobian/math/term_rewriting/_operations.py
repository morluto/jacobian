"""Domain adapter for term rewriting operations."""

from __future__ import annotations

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
    rewrite_step,
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
        return UnificationResult(unified=False, substitution={})
    return UnificationResult(unified=True, substitution=result)


def compute_rewrite_step(request: RewriteStepRequest) -> RewriteStepResult:
    rewritten, term = rewrite_step(request.term, list(request.rules))
    return RewriteStepResult(rewritten=rewritten, term=term)


def compute_normal_form(request: NormalFormRequest) -> NormalFormResult:
    term, converged, steps = normal_form(
        request.term, list(request.rules), request.max_steps,
    )
    return NormalFormResult(term=term, converged=converged, steps=steps)
