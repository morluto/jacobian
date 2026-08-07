from __future__ import annotations

import importlib

import pytest

OPERATION_MODULES = (
    "jacobian.domains.arithmetic.operations",
    "jacobian.domains.combinatorics.operations",
    "jacobian.domains.finite_sets.operations",
    "jacobian.domains.geometry.operations",
    "jacobian.domains.graph_optimization.operations",
    "jacobian.domains.matrix_lattice.operations",
    "jacobian.domains.number_theory.operations",
    "jacobian.domains.polynomial.operations",
    "jacobian.domains.sequences.operations",
)


@pytest.mark.parametrize("module_name", OPERATION_MODULES)
def test_operation_modules_export_exactly_their_defined_public_symbols(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    defined_public_symbols = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
    }

    assert module.__all__ == sorted(defined_public_symbols)
