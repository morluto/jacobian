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
from pathlib import Path

import jacobian

ROOT_MATH_DOMAINS = (
    "analysis",
    "cluster_algebras",
    "coalgebras",
    "combinatorics",
    "crossed_products",
    "dynamics",
    "finite_categories",
    "finite_dim_algebras",
    "finite_fields",
    "finite_semigroups",
    "geometry",
    "graphs",
    "groups",
    "lattices",
    "logic",
    "matrices",
    "number_theory",
    "optimization",
    "polynomials",
    "probability",
    "topology",
    "universal_algebra",
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


def test_public_math_namespaces_expose_no_wire_models() -> None:
    """Every declared native package excludes wire requests from its exports."""

    math_root = Path(__file__).parents[3] / "src" / "jacobian" / "math"
    modules = (
        importlib.import_module(
            "jacobian.math."
            + path.parent.relative_to(math_root).as_posix().replace("/", ".")
        )
        for path in math_root.rglob("__init__.py")
        if path.parent != math_root
    )
    for module in modules:
        assert all(
            not name.endswith(("Request", "Input"))
            for name in getattr(module, "__all__", ())
        ), f"{module.__name__} exports a wire model"


def test_public_values_and_functions_have_one_canonical_module() -> None:
    """Every exported value or callable has one owner across the full math tree."""

    math_root = Path(__file__).parents[3] / "src" / "jacobian" / "math"
    modules = tuple(
        "jacobian.math."
        + path.parent.relative_to(math_root).as_posix().replace("/", ".")
        for path in math_root.rglob("__init__.py")
        if path.parent != math_root
    )
    function_locations: dict[object, list[str]] = {}
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            if callable(value) and not isinstance(value, type(importlib)):
                function_locations.setdefault(value, []).append(f"{module_name}.{name}")
    assert all(len(locations) == 1 for locations in function_locations.values())


def test_root_namespace_stays_minimal() -> None:
    assert jacobian.__all__ == []
    assert not hasattr(jacobian, "VerificationResult")


def test_parallel_contract_and_domain_namespaces_are_deleted() -> None:
    import pytest

    for module_name in ("jacobian.contracts", "jacobian.domains"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_public_math_exports_are_resolvable() -> None:
    """Every symbol declared by a public math owner must be resolvable."""

    from jacobian import math

    for domain in math.__all__:
        module = importlib.import_module(f"jacobian.math.{domain}")
        assert hasattr(module, "__all__"), f"{domain} has no __all__"
        for name in module.__all__:
            assert hasattr(module, name), f"{domain}.{name} is missing"
