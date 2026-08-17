"""Explicit immutable inventory of built-in mathematical tool modules."""

from __future__ import annotations

from importlib import import_module
from typing import cast

from jacobian.math_tools import MathTools

type BuiltinOperationModule = tuple[str, str]
type LoadedOperationModule = tuple[str, MathTools]

BUILTIN_OPERATION_MODULES: tuple[BuiltinOperationModule, ...] = (
    ("jacobian.domains.boolean", "boolean_operations"),
    ("jacobian.domains.group", "group_operations"),
    ("jacobian.domains.graph_coloring_ops", "graph_coloring_operations"),
    ("jacobian.domains.graph_spectral", "graph_spectral_operations"),
    ("jacobian.domains.graph_flow", "graph_flow_operations"),
    (
        "jacobian.domains.graph_decomposition",
        "graph_decomposition_operations",
    ),
    ("jacobian.domains.graph_isomorphism", "graph_isomorphism_operations"),
    ("jacobian.domains.root_isolation", "root_isolation_operations"),
    ("jacobian.domains.recurrence_solving", "recurrence_solving_operations"),
    ("jacobian.domains.code_theory", "code_theory_operations"),
    ("jacobian.domains.number_field", "number_field_operations"),
    ("jacobian.domains.markov_chain", "markov_chain_operations"),
    ("jacobian.domains.arithmetic", "arithmetic_operations"),
    ("jacobian.domains.number_theory", "number_theory_operations"),
    ("jacobian.domains.combinatorics", "combinatorics_operations"),
    ("jacobian.domains.finite_sets", "finite_set_operations"),
    ("jacobian.domains.finite_fields", "finite_field_operations"),
    ("jacobian.domains.logic", "logic_operations"),
    ("jacobian.domains.sequences", "sequence_operations"),
    ("jacobian.domains.geometry", "geometry_operations"),
    (
        "jacobian.domains.projective_geometry",
        "projective_geometry_operations",
    ),
    (
        "jacobian.domains.graph_optimization",
        "graph_optimization_operations",
    ),
    (
        "jacobian.domains.graph_optimization",
        "graph_invariant_operations",
    ),
    (
        "jacobian.domains.graph_symmetry",
        "graph_symmetry_operations",
    ),
    ("jacobian.domains.certified_snf", "certified_snf_operations"),
    ("jacobian.domains.matrices", "matrix_operations"),
    ("jacobian.domains.symbolic_matrix", "symbolic_matrix_operations"),
    (
        "jacobian.domains.rational_linear",
        "rational_linear_operations",
    ),
    ("jacobian.domains.lattices", "lattice_operations"),
    ("jacobian.domains.formal_power_series", "formal_power_series_operations"),
    ("jacobian.domains.polynomial", "polynomial_operations"),
    (
        "jacobian.domains.multivariate_polynomial",
        "multivariate_polynomial_operations",
    ),
    ("jacobian.domains.analysis", "real_analysis_operations"),
    (
        "jacobian.domains.probability",
        "finite_probability_operations",
    ),
    (
        "jacobian.domains.optimization",
        "rational_optimization_operations",
    ),
    ("jacobian.domains.topology", "topology_operations"),
    ("jacobian.domains.posets", "finite_poset_operations"),
    ("jacobian.domains.directed_graph", "directed_graph_operations"),
    (
        "jacobian.domains.discrepancy_theory",
        "discrepancy_theory_operations",
    ),
    ("jacobian.domains.graph_polynomials", "graph_polynomial_operations"),
    ("jacobian.domains.numerical_semigroups", "numerical_semigroup_operations"),
    ("jacobian.domains.multiple_testing", "multiple_testing_operations"),
    ("jacobian.domains.exact_geometry", "exact_geometry_operations"),
    ("jacobian.domains.arithmetic_counting", "arithmetic_counting_operations"),
    (
        "jacobian.domains.graph_realization",
        "graph_realization_operations",
    ),
    (
        "jacobian.domains.boolean_analysis",
        "boolean_analysis_operations",
    ),
    (
        "jacobian.domains.arithmetic_functions",
        "arithmetic_functions_operations",
    ),
    (
        "jacobian.domains.additive_combinatorics",
        "additive_combinatorics_operations",
    ),
    ("jacobian.domains.matrix_analysis", "matrix_analysis_operations"),
    ("jacobian.domains.convex_analysis", "convex_analysis_operations"),
    ("jacobian.domains.submodular_opt", "submodular_opt_operations"),
    ("jacobian.domains.polynomial_maps", "polynomial_map_operations"),
    ("jacobian.domains.euclidean_geometry", "euclidean_geometry_operations"),
    ("jacobian.domains.finite_game_theory", "finite_game_theory_operations"),
)


def load_builtin_operation_modules() -> tuple[LoadedOperationModule, ...]:
    """Load the fixed built-in inventory for catalog compilation or binding."""

    return tuple(
        load_builtin_operation_module(module_name, factory_name)
        for module_name, factory_name in BUILTIN_OPERATION_MODULES
    )


def load_builtin_operation_module(
    module_name: str,
    factory_name: str | None = None,
) -> LoadedOperationModule:
    """Load one selected module from the fixed built-in inventory."""

    matching_factories = tuple(
        factory
        for configured_module_name, factory in BUILTIN_OPERATION_MODULES
        if configured_module_name == module_name
    )

    if factory_name is None:
        if not matching_factories:
            raise ValueError(f"unknown built-in operation module: {module_name}")
        if len(matching_factories) > 1:
            duplicate_factories = ", ".join(matching_factories)
            raise ValueError(
                "ambiguous built-in operation module; specify factory with one of: "
                f"{duplicate_factories}",
            )
        factory_name = matching_factories[0]
    elif factory_name not in matching_factories:
        raise ValueError(
            f"unknown built-in operation module/factory: {module_name}.{factory_name}",
        )
    module = import_module(module_name)
    factory = getattr(module, factory_name)
    operations = cast(MathTools, factory())
    return module_name, operations


__all__ = [
    "BUILTIN_OPERATION_MODULES",
    "load_builtin_operation_module",
    "load_builtin_operation_modules",
]
