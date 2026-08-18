"""Owner-local exact public API contract for formal_power_series."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.formal_power_series")
    expected = (
        "TruncatedSeries",
        "add",
        "compose",
        "derivative",
        "divide",
        "from_polynomial",
        "identity_check",
        "integral_zero_constant",
        "inverse",
        "multiply",
        "power",
        "reversion",
        "scalar_multiply",
        "subtract",
        "to_polynomial",
        "truncate",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
