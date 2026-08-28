"""Exact public API contract for jacobian.math.dynamics.arithmetic."""

from __future__ import annotations

from jacobian.math.dynamics import arithmetic as arithmetic_dynamics


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the arithmetic_dynamics public API."""
    expected = (
        "FunctionalGraph",
        "OrbitComputation",
        "RepeatEvidence",
        "cycle_multiplier",
        "dynatomic_polynomial",
        "finite_field_functional_graph",
        "fixed_point_equation",
        "iterate_polynomial",
        "orbit_prefix",
        "polynomial_coefficients",
        "polynomial_from_coefficients",
        "validate_cycle",
    )
    assert tuple(arithmetic_dynamics.__all__) == expected
    assert len(arithmetic_dynamics.__all__) == len(set(arithmetic_dynamics.__all__))
    assert all(not name.startswith("_") for name in arithmetic_dynamics.__all__)
    assert all(
        hasattr(arithmetic_dynamics, name) for name in arithmetic_dynamics.__all__
    )
