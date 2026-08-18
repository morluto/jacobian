"""Supported native first-order term-rewriting API."""

from jacobian.math.term_rewriting.operations import (
    apply_substitution,
    match,
    normal_form,
    rewrite_steps,
    selected_rewrite_step,
    term_at_position,
    unify,
)
from jacobian.math.term_rewriting.values import RewriteApplication, RewriteRule, Term

__all__ = [
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
]
