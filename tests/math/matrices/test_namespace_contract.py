"""Owner-local exact public API contract for matrices."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.matrices")
    expected = (
        "SmithNormalForm",
        "adjugate",
        "characteristic_polynomial",
        "determinant",
        "inverse",
        "kronecker_product",
        "multiply",
        "partial_trace",
        "permanent",
        "rank",
        "rref",
        "smith_normal_form",
        "solve_linear_system",
        "trace",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
