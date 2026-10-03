"""Catalog discovery check for the function-field operations."""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS

OPERATION_ID = "function_field.element.multiply.compute"


def test_catalog_discovery() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids
    assert "number_theory.function_field.multiply.compute" not in ids
