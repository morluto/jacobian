"""Exact public API contract for jacobian.math.polynomials.series."""

from __future__ import annotations

from jacobian.math.polynomials import series as formal_power_series


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the formal_power_series public API."""
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
    assert tuple(formal_power_series.__all__) == expected
    assert len(formal_power_series.__all__) == len(set(formal_power_series.__all__))
    assert all(not name.startswith("_") for name in formal_power_series.__all__)
    assert all(
        hasattr(formal_power_series, name) for name in formal_power_series.__all__
    )
