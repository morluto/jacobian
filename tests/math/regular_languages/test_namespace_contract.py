"""Owner-local exact public API contract for regular_languages."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.regular_languages")
    expected = (
        "DFA",
        "DFATransition",
        "count_accepted_words",
        "dfa_complement",
        "dfa_run",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
