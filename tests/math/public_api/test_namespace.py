from __future__ import annotations

import importlib

import pytest

import jacobian

PUBLIC_API = {
    "jacobian.math": (
        "algebraic_combinatorics",
        "arithmetic",
        "combinatorics",
        "diophantine_approximation",
        "finite_abelian_groups",
        "finite_fields",
        "finite_metric_spaces",
        "formal_power_series",
        "graphs",
        "matrices",
        "numerical_semigroups",
        "petri_nets",
        "polynomials",
        "prime_field_linear_algebra",
        "probability",
<<<<<<< HEAD
        "regular_languages",
    ),
    "jacobian.math.algebraic_combinatorics": (
        "conjugate_partition",
        "hook_lengths",
        "standard_young_tableaux_count",
=======
        "term_rewriting",
>>>>>>> 344b98e66 (fix term rewriting public capability boundary)
    ),
    "jacobian.math.finite_fields": (
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
    "jacobian.math.combinatorics": (
        "IndexedRecurrenceResidual",
        "PolynomialCoefficientRecurrenceTableRequest",
        "PolynomialCoefficientRecurrenceTableResult",
        "bell_number",
        "bernoulli_number",
        "catalan_number",
        "derangement_number",
        "double_factorial",
        "fibonacci_number",
        "integer_partitions",
        "lucas_number",
        "motzkin_number",
        "partition_number",
        "recurrence_table_residuals",
        "stirling_first",
        "stirling_second",
    ),
    "jacobian.math.diophantine_approximation": (
        "continued_fraction",
        "convergents",
        "solve_pell",
    ),
    "jacobian.math.finite_metric_spaces": (
        "ball",
        "gromov_hyperbolicity",
        "metric_profile",
    ),
    "jacobian.math.formal_power_series": (
        "TruncatedSeries",
        "add",
        "compose",
        "derivative",
        "divide",
        "from_polynomial",
        "identity_check",
        "integral_zero_constant",
        "inverse",
        "multiply",
        "power",
        "reversion",
        "scalar_multiply",
        "subtract",
        "to_polynomial",
        "truncate",
    ),
    "jacobian.math.graphs": (
        "GraphCompositionInput",
        "IndependenceNumberBudget",
        "IndependenceNumberRequest",
        "IndependenceNumberResult",
        "SimpleUndirectedGraph",
        "biconnected_components",
        "complement",
        "compose_graphs",
        "diameter",
        "explicit_graph",
        "graph_power",
        "independence_number",
        "induced_subgraph",
        "is_eulerian",
        "line_graph",
        "radius",
        "strongly_connected_components",
        "triangle_count",
    ),
    "jacobian.math.matrices": (
        "SmithNormalForm",
        "adjugate",
        "characteristic_polynomial",
        "determinant",
        "inverse",
        "kronecker_product",
        "multiply",
        "partial_trace",
        "permanent",
        "rank",
        "rref",
        "smith_normal_form",
        "solve_linear_system",
        "trace",
    ),
    "jacobian.math.numerical_semigroups": (
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
    ),
    "jacobian.math.petri_nets": (
        "Marking",
        "PetriNet",
        "compute_incidence_matrix",
        "enabled_transitions",
        "fire_transition",
        "reachability_graph",
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
<<<<<<< HEAD
    "jacobian.math.regular_languages": (
        "DFA",
        "DFATransition",
        "count_accepted_words",
        "dfa_complement",
        "dfa_run",
=======
    "jacobian.math.term_rewriting": (
        "RewriteApplication",
        "RewriteRule",
        "Term",
        "apply_substitution",
        "match",
        "normal_form",
        "rewrite_steps",
        "selected_rewrite_step",
        "term_at_position",
        "unify",
>>>>>>> 344b98e66 (fix term rewriting public capability boundary)
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


def test_parallel_contract_and_domain_namespaces_are_deleted() -> None:
    for module_name in ("jacobian.contracts", "jacobian.domains"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
