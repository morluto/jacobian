"""Catalog discovery check for the divisibility-poset operations."""

from __future__ import annotations

from jacobian.catalog.builtins import BUILTIN_TOOLS


def test_catalog_discovery() -> None:
    ids = [tool.operation_id for tool in BUILTIN_TOOLS]
    assert "integer.divisibility_poset.compute" in ids
    assert "number_theory.divisibility_poset.compute" not in ids
