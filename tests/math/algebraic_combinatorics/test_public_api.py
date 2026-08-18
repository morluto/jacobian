"""Exact public API contract for jacobian.math.algebraic_combinatorics."""

from __future__ import annotations

from jacobian.math import algebraic_combinatorics


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the algebraic_combinatorics public API."""
    expected = (
        "conjugate_partition",
        "hook_lengths",
        "standard_young_tableaux_count",
    )
    assert tuple(algebraic_combinatorics.__all__) == expected
    assert len(algebraic_combinatorics.__all__) == len(
        set(algebraic_combinatorics.__all__)
    )
    assert all(not name.startswith("_") for name in algebraic_combinatorics.__all__)
    assert all(
        hasattr(algebraic_combinatorics, name)
        for name in algebraic_combinatorics.__all__
    )
