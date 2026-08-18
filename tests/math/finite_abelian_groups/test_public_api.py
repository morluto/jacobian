"""Exact public API contract for jacobian.math.finite_abelian_groups."""

from __future__ import annotations

from jacobian.math import finite_abelian_groups


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the finite_abelian_groups public API."""
    expected = (
        "FiniteAbelianGroupFactorizationRequest",
        "FiniteAbelianGroupFactorizationResult",
        "FiniteAbelianRepresentationCount",
        "FiniteAbelianRepresentationWitness",
        "finite_abelian_group_factorization",
    )
    assert tuple(finite_abelian_groups.__all__) == expected
    assert len(finite_abelian_groups.__all__) == len(set(finite_abelian_groups.__all__))
    assert all(not name.startswith("_") for name in finite_abelian_groups.__all__)
    assert all(
        hasattr(finite_abelian_groups, name) for name in finite_abelian_groups.__all__
    )
