"""Exact public API contract for jacobian.math.term_rewriting."""

from __future__ import annotations

from jacobian.math import term_rewriting


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the term_rewriting public API."""
    expected = (
        "RewriteApplication",
        "RewriteRule",
        "Term",
        "apply_substitution",
        "match",
        "normal_form",
        "rewrite_steps",
        "selected_rewrite_step",
        "term_at_position",
        "unify",
    )
    assert tuple(term_rewriting.__all__) == expected
    assert len(term_rewriting.__all__) == len(set(term_rewriting.__all__))
    assert all(not name.startswith("_") for name in term_rewriting.__all__)
    assert all(hasattr(term_rewriting, name) for name in term_rewriting.__all__)
