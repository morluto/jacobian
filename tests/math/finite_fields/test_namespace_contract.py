"""Owner-local exact public API contract for finite_fields."""

from __future__ import annotations

import importlib


def test_public_manifest_is_exact() -> None:
    module = importlib.import_module("jacobian.math.finite_fields")
    expected = (
        "Axis",
        "AxisBoundMatrix",
        "CollisionResult",
        "DirectionRankLedger",
        "FiberPartition",
        "FiniteDimensionalSubspace",
        "FiniteFieldElement",
        "FiniteFieldPresentation",
        "FiniteLinearMap",
        "FiniteMapTable",
        "FinitePolynomial",
        "FinitePolynomialMap",
        "OrbitDistribution",
        "PermutationResult",
        "ProjectiveLine",
        "ProjectivePoint",
        "RankResult",
        "analyze_collisions",
        "analyze_permutation",
        "direction_rank_ledger",
        "element",
        "evaluate_finite_polynomial",
        "fiber_partition",
        "finite_field",
        "finite_map_table",
        "finite_polynomial",
        "finite_polynomial_map",
        "linear_map_rank",
        "orbit_distribution",
        "projective_line",
        "projective_point",
        "restrict_scalars",
    )
    assert tuple(module.__all__) == expected
    assert len(expected) == len(set(expected))
    assert all(not name.startswith("_") for name in expected)
    assert all(hasattr(module, name) for name in expected)
