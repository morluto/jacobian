from __future__ import annotations

import importlib

import pytest

import jacobian

PUBLIC_API = {
    "jacobian.math": (
        "arithmetic",
        "finite_abelian_groups",
        "finite_fields",
        "graphs",
        "matrices",
        "polynomials",
        "prime_field_linear_algebra",
        "probability",
    ),
    "jacobian.math.finite_fields": (
        "Axis",
        "AxisBoundMatrix",
        "CollisionCertificate",
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
        "PermutationCertificate",
        "ProjectiveLine",
        "ProjectivePoint",
        "RankResult",
        "collision_certificate",
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
        "permutation_certificate",
        "projective_line",
        "projective_point",
        "restrict_scalars",
    ),
    "jacobian.math.arithmetic": (
        "absolute_value",
        "integerize_rational_vector",
        "primitive_integer_vector",
        "quotient",
        "reciprocal",
        "sign",
        "sum_rationals",
    ),
    "jacobian.math.graphs": (
        "GraphCompositionInput",
        "IndependenceNumberBudget",
        "IndependenceNumberRequest",
        "IndependenceNumberResult",
        "SimpleUndirectedGraph",
        "compose_graphs",
        "diameter",
        "explicit_graph",
        "independence_number",
        "is_eulerian",
        "triangle_count",
    ),
    "jacobian.math.matrices": (
        "SmithNormalForm",
        "adjugate",
        "characteristic_polynomial",
        "determinant",
        "inverse",
        "multiply",
        "rank",
        "rref",
        "smith_normal_form",
        "solve_linear_system",
        "trace",
    ),
    "jacobian.math.polynomials": (
        "derivative",
        "discriminant",
        "divide",
        "evaluate",
        "factorization",
        "gcdex",
        "groebner_basis",
        "integral",
        "partial_fractions",
        "resultant",
        "square_free_decomposition",
    ),
    "jacobian.math.prime_field_linear_algebra": (
        "PrimeFieldMatrix",
        "column_basis",
        "nullspace",
        "quotient_basis",
        "rank",
        "rref",
    ),
    "jacobian.math.probability": (
        "FiniteJointTable",
        "MutualInformationCertificate",
        "MutualInformationResult",
        "MutualInformationTerm",
        "mutual_information",
    ),
}


def test_public_manifest_is_exact() -> None:
    for module_name, names in PUBLIC_API.items():
        module = importlib.import_module(module_name)
        assert tuple(module.__all__) == names
        assert len(names) == len(set(names))
        assert all(not name.startswith("_") for name in names)
        assert all(hasattr(module, name) for name in names)


def test_functions_have_one_canonical_module() -> None:
    function_locations: dict[object, list[str]] = {}
    for module_name, names in PUBLIC_API.items():
        module = importlib.import_module(module_name)
        for name in names:
            value = getattr(module, name)
            if callable(value) and not isinstance(value, type(importlib)):
                function_locations.setdefault(value, []).append(f"{module_name}.{name}")
    assert all(len(locations) == 1 for locations in function_locations.values())


def test_root_namespace_stays_minimal() -> None:
    assert jacobian.__all__ == []
    assert not hasattr(jacobian, "VerificationResult")


def test_shared_contract_namespace_contains_only_passive_primitives() -> None:
    contracts = importlib.import_module("jacobian.contracts")

    assert contracts.__all__ == [
        "ArtifactUri",
        "CheckerUri",
        "Sha256Digest",
        "ValueUri",
    ]
    assert not hasattr(contracts, "VerificationResult")


def test_deleted_experimental_contract_modules_are_not_importable() -> None:
    for module_name in (
        "jacobian.contracts.proof_ir",
        "jacobian.contracts.lean_artifacts",
        "jacobian.contracts.plugin_inputs",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
