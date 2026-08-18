"""Exact public API contract for jacobian.math.diophantine_approximation."""

from __future__ import annotations

from jacobian.math import diophantine_approximation


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the diophantine_approximation public API."""
    expected = (
        "continued_fraction",
        "convergents",
        "solve_pell",
    )
    assert tuple(diophantine_approximation.__all__) == expected
    assert len(diophantine_approximation.__all__) == len(
        set(diophantine_approximation.__all__)
    )
    assert all(not name.startswith("_") for name in diophantine_approximation.__all__)
    assert all(
        hasattr(diophantine_approximation, name)
        for name in diophantine_approximation.__all__
    )
