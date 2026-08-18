"""Owner-local exact public API contract for prime_field_linear_algebra."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.prime_field_linear_algebra")
    expected = (
        "PrimeFieldMatrix",
        "column_basis",
        "nullspace",
        "quotient_basis",
        "rank",
        "rref",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
