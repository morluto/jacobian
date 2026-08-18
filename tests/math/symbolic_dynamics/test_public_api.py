"""Exact public API contract for jacobian.math.symbolic_dynamics."""

from __future__ import annotations

from jacobian.math import symbolic_dynamics


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the symbolic_dynamics public API."""
    expected = (
        "AdjacencyShift",
        "BlockPresentation",
        "ForbiddenBlockShift",
        "LabeledTransition",
        "adjacency_shift",
        "block_language",
        "finite_type_presentation",
        "higher_block_presentation",
        "normalize_forbidden_blocks",
        "periodic_point_profile",
    )
    assert tuple(symbolic_dynamics.__all__) == expected
    assert len(symbolic_dynamics.__all__) == len(set(symbolic_dynamics.__all__))
    assert all(not name.startswith("_") for name in symbolic_dynamics.__all__)
    assert all(hasattr(symbolic_dynamics, name) for name in symbolic_dynamics.__all__)
