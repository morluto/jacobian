"""Exact public API contract for jacobian.math.graphical_models."""

from __future__ import annotations

from jacobian.math import graphical_models


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the graphical_models public API."""
    expected = (
        "Factor",
        "d_separation",
        "factor_marginalize",
        "factor_multiply",
        "variable_elimination",
    )
    assert tuple(graphical_models.__all__) == expected
    assert len(graphical_models.__all__) == len(set(graphical_models.__all__))
    assert all(not name.startswith("_") for name in graphical_models.__all__)
    assert all(hasattr(graphical_models, name) for name in graphical_models.__all__)
