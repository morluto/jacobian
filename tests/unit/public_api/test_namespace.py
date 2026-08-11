from __future__ import annotations

import importlib

import pytest

import jacobian

PUBLIC_API = {
    "jacobian.math": (
        "arithmetic",
        "graphs",
        "matrices",
        "polynomials",
        "prime_field_linear_algebra",
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
    "jacobian.math.graphs": ("diameter", "is_eulerian", "triangle_count"),
    "jacobian.math.matrices": ("determinant", "inverse", "rank", "rref", "trace"),
    "jacobian.math.polynomials": ("derivative", "gcdex", "resultant"),
    "jacobian.math.prime_field_linear_algebra": (
        "PrimeFieldMatrix",
        "column_basis",
        "nullspace",
        "quotient_basis",
        "rank",
        "rref",
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
    assert jacobian.__all__ == ["ResultEnvelope"]
    assert "arithmetic" not in jacobian.__all__
    assert "matrices" not in jacobian.__all__
    assert "graphs" not in jacobian.__all__
    assert "polynomials" not in jacobian.__all__


def test_deleted_experimental_contract_modules_are_not_importable() -> None:
    for module_name in (
        "jacobian.contracts.proof_ir",
        "jacobian.contracts.lean_artifacts",
        "jacobian.contracts.plugin_inputs",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_unsupported_aliases_are_not_exposed() -> None:
    operations = importlib.import_module("jacobian.domains.probability.operations")
    assert hasattr(operations, "FINITE_PROBABILITY_CAPABILITIES")
    assert not hasattr(operations, "FINITE_MOMENT_CAPABILITIES")
