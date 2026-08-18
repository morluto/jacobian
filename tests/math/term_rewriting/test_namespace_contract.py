"""Owner-local exact public API contract for term_rewriting."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.term_rewriting")
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
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
