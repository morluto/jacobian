"""Exact public API contract for jacobian.math.number_theory.diophantine_approximation."""

from __future__ import annotations

from jacobian.math.number_theory import diophantine_approximation


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the diophantine_approximation public API."""
    expected = (
        "continued_fraction",
        "convergents",
        "nearest_integer_distance",
        "range_profile",
        "record_minima",
        "scaled_floor",
        "simultaneous_product",
        "solve_pell",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(diophantine_approximation.__all__)
    assert len(diophantine_approximation.__all__) == len(
        set(diophantine_approximation.__all__)
    )
    assert all(not name.startswith("_") for name in diophantine_approximation.__all__)
    assert all(
        hasattr(diophantine_approximation, name)
        for name in diophantine_approximation.__all__
    )
