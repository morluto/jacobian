"""Supported native first-order term-rewriting operations."""

from jacobian.math.logic.term_rewriting._kernel import (
    apply_substitution,
    critical_pairs,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
    unify,
)

__all__ = [
    "apply_substitution",
    "critical_pairs",
    "match",
    "normal_form",
    "rewrite_steps",
    "selected_rewrite_step",
    "term_at_position",
    "unify",
]
