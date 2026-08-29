"""Supported native first-order term-rewriting operations."""

from typing import Literal

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.term_rewriting._kernel import (
    _bounded_unify,
    _critical_pairs,
    _validate_critical_pair_source,
    apply_substitution,
    critical_pairs,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
    unify,
)
from jacobian.math.logic.term_rewriting._models import (
    CriticalPairsResult,
    MatchingResult,
    NormalFormResult,
    RewriteStepResult,
    RewriteStepSelection,
    SubstitutionResult,
    UnificationResult,
)
from jacobian.math.logic.term_rewriting.values import (
    RankedSignature,
    RewriteRule,
    Substitution,
    Term,
    _require_transport_safe_depth,
)

__all__ = [
    "apply_substitution",
    "critical_pairs",
    "critical_pairs_result",
    "match",
    "matching_result",
    "normal_form",
    "normal_form_result",
    "rewrite_step_result",
    "rewrite_steps",
    "selected_rewrite_step",
    "substitution_result",
    "term_at_position",
    "unification_result",
    "unify",
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


def substitution_result(
    signature: RankedSignature,
    term: Term,
    substitution: Substitution,
) -> SubstitutionResult:
    """Apply a substitution to a term after signature and depth admission."""

    replacements = tuple(substitution.mapping.values())
    _admit_terms(signature, (term, *replacements), location=("term",))
    result = apply_substitution(term, substitution.mapping)
    try:
        _require_transport_safe_depth(result)
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="transport_depth", location=("substitution",)
        ) from error
    return SubstitutionResult._from_kernel(
        signature=signature,
        term=term,
        substitution=substitution,
        result=result,
    )


def matching_result(
    signature: RankedSignature,
    pattern: Term,
    subject: Term,
) -> MatchingResult:
    """Match a canonical pattern and subject after signature admission."""

    _admit_terms(signature, (pattern, subject), location=("pattern",))
    result = match(pattern, subject)
    if result is None:
        return MatchingResult._from_kernel(
            signature=signature,
            pattern=pattern,
            subject=subject,
            matched=False,
            substitution={},
        )
    return MatchingResult._from_kernel(
        signature=signature,
        pattern=pattern,
        subject=subject,
        matched=True,
        substitution=result,
    )


def unification_result(
    signature: RankedSignature,
    left: Term,
    right: Term,
) -> UnificationResult:
    """Unify canonical terms after signature and result-depth admission."""

    _admit_terms(signature, (left, right), location=("left",))
    try:
        result = _bounded_unify(left, right)
        if result is not None:
            _require_transport_safe_depth(*result.values())
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="unification_bound", location=("left", "right")
        ) from error
    if result is None:
        return UnificationResult._from_kernel(
            signature=signature,
            left=left,
            right=right,
            unified=False,
            substitution={},
        )
    return UnificationResult._from_kernel(
        signature=signature,
        left=left,
        right=right,
        unified=True,
        substitution=result,
    )


def rewrite_step_result(
    signature: RankedSignature,
    term: Term,
    rules: tuple[RewriteRule, ...],
    selection: RewriteStepSelection | None,
) -> RewriteStepResult:
    """Enumerate or select one rewrite step for canonical source values."""

    _admit_terms(
        signature,
        (term, *_rule_terms(rules)),
        location=("term", "rules"),
    )
    scope: Literal["ALL_APPLICABLE_STEPS", "SELECTED_STEP"]
    try:
        if selection is None:
            applications = rewrite_steps(term, rules)
            scope = "ALL_APPLICABLE_STEPS"
        else:
            if selection.rule_index >= len(rules):
                raise ValueError("selected rule_index is out of range")
            term_at_position(term, selection.position)
            application = selected_rewrite_step(
                term,
                rules,
                selection.position,
                selection.rule_index,
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
    return RewriteStepResult._from_kernel(
        signature=signature,
        source_term=term,
        rules=rules,
        selection=selection,
        scope=scope,
        applications=applications,
    )


def normal_form_result(
    signature: RankedSignature,
    term: Term,
    rules: tuple[RewriteRule, ...],
    strategy: Literal["LEFTMOST_OUTERMOST_RULE_ORDER"],
    max_steps: int,
) -> NormalFormResult:
    """Run bounded canonical normalization after signature admission."""

    source_term = term
    _admit_terms(
        signature,
        (term, *_rule_terms(rules)),
        location=("term", "rules"),
    )
    try:
        term, status, steps, next_step = normal_form(term, rules, max_steps)
        observed = (term,) if next_step is None else (term, next_step.term)
        _require_transport_safe_depth(*observed)
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="normal_form_bound", location=("term", "rules")
        ) from error
    return NormalFormResult._from_kernel(
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


def critical_pairs_result(
    signature: RankedSignature,
    rules: tuple[RewriteRule, ...],
) -> CriticalPairsResult:
    """Compute the complete critical-pair profile for canonical rule values."""

    try:
        _validate_critical_pair_source(signature, rules)
        _require_transport_safe_depth(*_rule_terms(rules))
    except ValueError as error:
        raise _domain_error(
            error, fallback_code="critical_pair_source", location=("rules",)
        ) from error
    return CriticalPairsResult._from_kernel(
        signature=signature,
        rules=rules,
        profile=_critical_pairs(signature, rules),
    )
