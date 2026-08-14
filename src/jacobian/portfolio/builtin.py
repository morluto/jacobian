"""Explicit inventory of built-in mathematical operation modules."""

from __future__ import annotations

from importlib import import_module
from typing import cast

from jacobian.checker_operations import ExactReplayCheckerDeclaration
from jacobian.operation_declarations import OperationDeclarations

type BuiltinOperationModule = tuple[str, str]
type LoadedOperationModule = tuple[
    str,
    OperationDeclarations,
    tuple[ExactReplayCheckerDeclaration, ...],
]

BUILTIN_OPERATION_MODULES: tuple[BuiltinOperationModule, ...] = (
    ("jacobian.domains.arithmetic.bundle", "build_arithmetic_bundle"),
    ("jacobian.domains.number_theory.bundle", "build_number_theory_bundle"),
    ("jacobian.domains.combinatorics.bundle", "build_combinatorics_bundle"),
    ("jacobian.domains.finite_sets.bundle", "build_finite_set_bundle"),
    ("jacobian.domains.finite_fields.bundle", "build_finite_field_bundle"),
    ("jacobian.domains.formal_datasets.bundle", "build_formal_dataset_bundle"),
    ("jacobian.domains.sequences.bundle", "build_sequence_bundle"),
    ("jacobian.domains.geometry.bundle", "build_geometry_bundle"),
    (
        "jacobian.domains.projective_geometry.bundle",
        "build_projective_geometry_bundle",
    ),
    (
        "jacobian.domains.graph_optimization.bundle",
        "build_graph_optimization_bundle",
    ),
    (
        "jacobian.domains.graph_optimization.invariant_bundle",
        "build_graph_invariant_bundle",
    ),
    ("jacobian.domains.graph_symmetry.bundle", "build_graph_symmetry_bundle"),
    ("jacobian.domains.certified_snf.bundle", "build_certified_snf_bundle"),
    ("jacobian.domains.matrix_lattice.bundle", "build_matrix_bundle"),
    ("jacobian.domains.rational_linear.bundle", "build_rational_linear_bundle"),
    ("jacobian.domains.matrix_lattice.lattice_bundle", "build_lattice_bundle"),
    ("jacobian.domains.polynomial.bundle", "build_polynomial_bundle"),
    ("jacobian.domains.analysis.bundle", "build_real_analysis_bundle"),
    (
        "jacobian.domains.probability.bundle",
        "build_finite_probability_bundle",
    ),
    (
        "jacobian.domains.optimization.bundle",
        "build_rational_optimization_bundle",
    ),
    ("jacobian.domains.topology.bundle", "build_topology_bundle"),
    ("jacobian.domains.posets.bundle", "build_finite_poset_bundle"),
)


def load_builtin_operation_modules() -> tuple[LoadedOperationModule, ...]:
    """Load the fixed built-in inventory for catalog compilation or binding."""

    loaded: list[LoadedOperationModule] = []
    for module_name, factory_name in BUILTIN_OPERATION_MODULES:
        module = import_module(module_name)
        factory = getattr(module, factory_name)
        operations = cast(OperationDeclarations, factory())
        checkers = cast(
            tuple[ExactReplayCheckerDeclaration, ...],
            getattr(module, "CHECKER_DECLARATIONS", ()),
        )
        loaded.append((module_name, operations, checkers))
    return tuple(loaded)


__all__ = ["BUILTIN_OPERATION_MODULES", "load_builtin_operation_modules"]
