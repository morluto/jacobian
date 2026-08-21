"""Exact public API contract for jacobian.math.finite_metric_spaces."""

from __future__ import annotations

from jacobian.math import finite_metric_spaces


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the finite_metric_spaces public API."""
    expected = (
        "ball",
        "gromov_hyperbolicity",
        "metric_profile",
    )
    assert tuple(finite_metric_spaces.__all__) == expected
    assert len(finite_metric_spaces.__all__) == len(set(finite_metric_spaces.__all__))
    assert all(not name.startswith("_") for name in finite_metric_spaces.__all__)
    assert all(
        hasattr(finite_metric_spaces, name) for name in finite_metric_spaces.__all__
    )
