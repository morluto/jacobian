"""Catalog discovery check for the group character operations."""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS

OPERATION_ID = "class_function.inner_product.compute"


def test_catalog_discovery() -> None:
    ids = {tool.operation_id for tool in BUILTIN_TOOLS}
    assert OPERATION_ID in ids
    assert "number_theory.character.inner_product.compute" not in ids
