"""Owner-local exact public API contract for diophantine_approximation."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.diophantine_approximation")
    expected = (
        "continued_fraction",
        "convergents",
        "solve_pell",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
