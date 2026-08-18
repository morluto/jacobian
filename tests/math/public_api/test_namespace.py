"""Cross-owner public API invariants for jacobian.math.

Per-domain exact __all__ expectations live in owner-local
``tests/math/<domain>/test_public_api.py`` files.
"""

from __future__ import annotations

import importlib

import pytest

import jacobian

ROOT_DOMAIN_EXPORTS = (
    "algebraic_combinatorics",
    "arithmetic",
    "arithmetic_dynamics",
    "combinatorics",
    "diophantine_approximation",
    "finite_abelian_groups",
    "finite_fields",
    "finite_metric_spaces",
    "finite_state_transducers",
    "finite_topology",
    "formal_power_series",
    "graphical_models",
    "graphs",
    "impartial_games",
    "matrices",
    "numerical_semigroups",
    "petri_nets",
    "polynomials",
    "prime_field_linear_algebra",
    "probability",
    "regular_languages",
    "symbolic_dynamics",
    "term_rewriting",
    "tree_automata",
    "words",
)


def test_root_namespace_exports_exact_domains() -> None:
    """The root jacobian.math __init__ exports exactly the supported domains."""
    module = importlib.import_module("jacobian.math")
    assert tuple(module.__all__) == ROOT_DOMAIN_EXPORTS
    assert len(ROOT_DOMAIN_EXPORTS) == len(set(ROOT_DOMAIN_EXPORTS))


def test_public_names_have_no_private_entries() -> None:
    """No reachable public __all__ contains private (underscore-prefixed) names."""
    module = importlib.import_module("jacobian.math")
    for name in module.__all__:
        assert not name.startswith("_"), f"private name in root: {name}"


def test_functions_have_one_canonical_module() -> None:
    """Every public callable resolves to exactly one canonical owner."""
    function_locations: dict[object, list[str]] = {}
    module = importlib.import_module("jacobian.math")
    for domain in module.__all__:
        domain_module = importlib.import_module(f"jacobian.math.{domain}")
        for name in domain_module.__all__:
            value = getattr(domain_module, name)
            if callable(value) and not isinstance(value, type(importlib)):
                function_locations.setdefault(value, []).append(
                    f"jacobian.math.{domain}.{name}"
                )
    assert all(len(locations) == 1 for locations in function_locations.values())


def test_root_namespace_stays_minimal() -> None:
    assert jacobian.__all__ == []
    assert not hasattr(jacobian, "VerificationResult")


def test_parallel_contract_and_domain_namespaces_are_deleted() -> None:
    for module_name in ("jacobian.contracts", "jacobian.domains"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_public_math_modules_do_not_import_catalog() -> None:
    """Public math modules must not import catalog publication layers."""

    module = importlib.import_module("jacobian.math")
    for domain in module.__all__:
        domain_module = importlib.import_module(f"jacobian.math.{domain}")
        for attr_name in domain_module.__all__:
            attr = getattr(domain_module, attr_name)
            if hasattr(attr, "__module__"):
                mod = attr.__module__ or ""
                assert "catalog" not in mod, (
                    f"{domain}.{attr_name} resolved to catalog module: {mod}"
                )
