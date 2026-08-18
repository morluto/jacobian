"""Owner-local exact public API contract for numerical_semigroups."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.numerical_semigroups")
    expected = (
        "FactorizationGraph",
        "apery_set",
        "belongs",
        "elasticity",
        "element_catenary_degree",
        "element_delta_set",
        "element_elasticity",
        "factorization_count",
        "factorization_distance",
        "factorization_graph",
        "factorization_lengths",
        "factorizations",
        "minimal_generating_system",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
