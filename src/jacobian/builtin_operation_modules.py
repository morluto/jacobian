"""Explicit immutable inventory of built-in operation declaration modules."""

from __future__ import annotations

from importlib import import_module
from typing import cast

from jacobian.operation_declarations import OperationDeclarations

type BuiltinOperationModule = tuple[str, str]
type LoadedOperationModule = tuple[str, OperationDeclarations]

BUILTIN_OPERATION_MODULES: tuple[BuiltinOperationModule, ...] = (
    ("jacobian.domains.arithmetic.domain_declarations", "arithmetic_operations"),
    ("jacobian.domains.number_theory.domain_declarations", "number_theory_operations"),
    ("jacobian.domains.combinatorics.domain_declarations", "combinatorics_operations"),
    ("jacobian.domains.finite_sets.domain_declarations", "finite_set_operations"),
    ("jacobian.domains.finite_fields.domain_declarations", "finite_field_operations"),
    ("jacobian.domains.logic.domain_declarations", "logic_operations"),
    ("jacobian.domains.sequences.domain_declarations", "sequence_operations"),
    ("jacobian.domains.geometry.domain_declarations", "geometry_operations"),
    (
        "jacobian.domains.projective_geometry.domain_declarations",
        "projective_geometry_operations",
    ),
    (
        "jacobian.domains.graph_optimization.domain_declarations",
        "graph_optimization_operations",
    ),
    (
        "jacobian.domains.graph_optimization.invariant_declarations",
        "graph_invariant_operations",
    ),
    (
        "jacobian.domains.graph_symmetry.domain_declarations",
        "graph_symmetry_operations",
    ),
    ("jacobian.domains.certified_snf.domain_declarations", "certified_snf_operations"),
    ("jacobian.domains.matrix_lattice.domain_declarations", "matrix_operations"),
    (
        "jacobian.domains.rational_linear.domain_declarations",
        "rational_linear_operations",
    ),
    ("jacobian.domains.matrix_lattice.lattice_declarations", "lattice_operations"),
    ("jacobian.domains.polynomial.domain_declarations", "polynomial_operations"),
    ("jacobian.domains.analysis.domain_declarations", "real_analysis_operations"),
    (
        "jacobian.domains.probability.domain_declarations",
        "finite_probability_operations",
    ),
    (
        "jacobian.domains.optimization.domain_declarations",
        "rational_optimization_operations",
    ),
    ("jacobian.domains.topology.domain_declarations", "topology_operations"),
    ("jacobian.domains.posets.domain_declarations", "finite_poset_operations"),
)


def load_builtin_operation_modules() -> tuple[LoadedOperationModule, ...]:
    """Load the fixed built-in inventory for catalog compilation or binding."""

    return tuple(
        load_builtin_operation_module(module_name)
        for module_name, _factory_name in BUILTIN_OPERATION_MODULES
    )


def load_builtin_operation_module(module_name: str) -> LoadedOperationModule:
    """Load one selected module from the fixed built-in inventory."""

    try:
        factory_name = dict(BUILTIN_OPERATION_MODULES)[module_name]
    except KeyError as exc:
        raise ValueError(f"unknown built-in operation module: {module_name}") from exc
    module = import_module(module_name)
    factory = getattr(module, factory_name)
    operations = cast(OperationDeclarations, factory())
    return module_name, operations


__all__ = [
    "BUILTIN_OPERATION_MODULES",
    "load_builtin_operation_module",
    "load_builtin_operation_modules",
]
