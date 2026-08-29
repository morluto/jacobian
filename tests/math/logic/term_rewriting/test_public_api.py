"""Exact public API contract for jacobian.math.logic.term_rewriting."""

from __future__ import annotations

from jacobian.math.logic import term_rewriting


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the term_rewriting public API."""
    expected = (
        "CriticalOverlapCandidate",
        "CriticalPair",
        "CriticalPairProfile",
        "RankedSignature",
        "RewriteApplication",
        "RewriteRule",
        "RewriteStepSelection",
        "Substitution",
        "Term",
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
    )
    assert tuple(term_rewriting.__all__) == expected
    assert len(term_rewriting.__all__) == len(set(term_rewriting.__all__))
    assert all(not name.startswith("_") for name in term_rewriting.__all__)
    assert all(hasattr(term_rewriting, name) for name in term_rewriting.__all__)
