"""Owner-local exact public API contract for arithmetic_dynamics."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.arithmetic_dynamics")
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
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
