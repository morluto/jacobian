"""Cross-owner invariants for the public ``jacobian.math`` namespace.

Exact symbol expectations for each domain live in owner-local
``test_public_api.py`` files under ``tests/math/<domain>/``.

This module retains only the repository-wide composition contract:
the exact small set/order of root ``jacobian.math`` domain exports
plus cross-owner checks that no private names, duplicate exports,
or compatibility aliases leak into the public surface.
"""

from __future__ import annotations

import importlib

import jacobian

ROOT_MATH_DOMAINS = (
    "additive_combinatorics",
    "algebraic_combinatorics",
    "approximation_theory",
    "arithmetic",
    "arithmetic_dynamics",
    "combinatorial_maps",
    "combinatorial_matrices",
    "combinatorics",
    "diophantine_approximation",
    "finite_abelian_groups",
    "finite_fields",
    "finite_metric_spaces",
    "finite_state_transducers",
    "finite_topology",
    "formal_concept_analysis",
    "formal_power_series",
    "graphical_models",
    "graphs",
    "greedoids",
    "hypergraphs",
    "impartial_games",
    "matrices",
    "numerical_semigroups",
    "petri_nets",
    "polynomial_support_geometry",
    "polynomials",
    "prime_field_linear_algebra",
    "probability",
    "regular_languages",
    "symbolic_dynamics",
    "term_rewriting",
    "tree_automata",
    "universal_algebra",
    "words",
)


def test_root_math_namespace_is_exact() -> None:
    """The root ``jacobian.math.__all__`` must match the expected domain list."""
    from jacobian import math

    assert tuple(math.__all__) == ROOT_MATH_DOMAINS
    assert len(math.__all__) == len(set(math.__all__))


def test_no_private_names_in_any_public_all() -> None:
    """Every public ``__all__`` must exclude private (underscore-prefixed) names."""
    from jacobian import math

    for domain in math.__all__:
        module = importlib.import_module(f"jacobian.math.{domain}")
        assert hasattr(module, "__all__"), f"{domain} has no __all__"
        assert all(not name.startswith("_") for name in module.__all__), (
            f"{domain} exports a private name"
        )


def test_no_duplicate_root_exports() -> None:
    """Root domain exports must be unique."""
    from jacobian import math

    assert len(math.__all__) == len(set(math.__all__))


def test_public_functions_have_one_canonical_module() -> None:
    """Every public callable must resolve to one canonical owner, not an alias."""
    from jacobian import math

    function_locations: dict[object, list[str]] = {}
    for domain in math.__all__:
        module = importlib.import_module(f"jacobian.math.{domain}")
        for name in module.__all__:
            value = getattr(module, name)
            if callable(value) and not isinstance(value, type(importlib)):
                function_locations.setdefault(value, []).append(
                    f"jacobian.math.{domain}.{name}"
                )
    assert all(len(locations) == 1 for locations in function_locations.values())


def test_root_namespace_stays_minimal() -> None:
    assert jacobian.__all__ == []
    assert not hasattr(jacobian, "VerificationResult")


def test_parallel_contract_and_domain_namespaces_are_deleted() -> None:
    import pytest

    for module_name in ("jacobian.contracts", "jacobian.domains"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_public_math_imports_have_no_catalog_side_effect() -> None:
    """Importing a public math module must not import catalog publication layers."""

    from jacobian import math

    for domain in math.__all__:
        module = importlib.import_module(f"jacobian.math.{domain}")
        assert hasattr(module, "__all__"), f"{domain} has no __all__"
        # Every declared symbol must be resolvable on the public module.
        for name in module.__all__:
            assert hasattr(module, name), f"{domain}.{name} is missing"
