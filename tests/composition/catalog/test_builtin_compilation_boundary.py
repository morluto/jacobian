"""Executable ownership checks for the built-in composition boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from jacobian.builtin_operation_modules import BUILTIN_OPERATION_MODULES
from jacobian.runtime.selected_families import (
    selected_family_catalog_installers,
    selected_family_specs,
)

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "src" / "jacobian"

_DELETED_CATALOG_PHASE_MODULES = (
    "catalog_foundations.py",
    "catalog_resources.py",
    "catalog_checkers.py",
)
_CATALOG_OWNER_MODULES = (
    SOURCE / "catalog_build.py",
    SOURCE / "catalog_operations.py",
)
_FORBIDDEN_DOMAIN_INSTALL_NAMES = frozenset(
    {
        "build_graph_composition_operations",
        "build_graph_isomorphism_operation",
        "build_graph_operations",
        "build_polynomial_operations",
        "install_cadical_operations",
        "install_finite_coverage",
        "install_lean_checkers",
        "install_lean_exploration_operations",
        "install_lean_statement_operations",
        "install_nullstellensatz_core",
        "install_polynomial_expression_checker",
        "install_polynomial_interval_operations",
        "install_polynomial_positivity_operations",
        "install_polynomial_system_operations",
        "install_polytope_checkers",
        "install_sat_assignment_checker",
        "install_sat_lrat_verifier",
        "install_sat_unsat_proof_checker",
        "install_singular_producer",
        "install_smt_unsat_proof_checker",
        "install_universal_algebra_operations",
        "PolytopeSeparationAdapter",
        "PolynomialSystemRationalSearchAdapter",
        "SatCnfMaterializationAdapter",
    }
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return tuple(modules)


def _imported_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            names.update(alias.name for alias in node.names)
    return frozenset(names)


def test_builtin_inventory_is_explicit_without_importing_domain_modules() -> None:
    modules = BUILTIN_OPERATION_MODULES
    assert modules, "expected explicit built-in operation modules"
    assert len(modules) == len(set(modules)), "duplicate operation modules"
    assert all(module.startswith("jacobian.domains.") for module, _factory in modules)
    for path in _CATALOG_OWNER_MODULES:
        assert not any(
            module.startswith("jacobian.domains.") for module in _imports(path)
        )
        assert _imported_names(path).isdisjoint(_FORBIDDEN_DOMAIN_INSTALL_NAMES)


def test_deleted_hand_listed_catalog_phase_modules_are_gone() -> None:
    for name in _DELETED_CATALOG_PHASE_MODULES:
        assert not (SOURCE / name).exists()


def test_family_catalog_hooks_are_indexed_from_selected_family_specs() -> None:
    specs = selected_family_specs()
    installers = selected_family_catalog_installers()
    assert tuple(installers) == tuple(spec.origin for spec in specs)
    assert all(origin.startswith("family:") for origin in installers)
    selected_ids = [spec.operation_ids for spec in specs]
    seen: set[str] = set()
    for operation_ids in selected_ids:
        assert seen.isdisjoint(operation_ids)
        seen.update(operation_ids)
    assert "selected_family_catalog_installers" in _imported_names(
        SOURCE / "catalog_build.py"
    )
    assert "selected_family_specs" in _imported_names(SOURCE / "catalog_build.py")
