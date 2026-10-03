"""Advertised local-series example through the published catalog."""

from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_newton_edge_roots_declared_catalog_example_executes() -> None:

    operation = Catalog.open().operation(
        "local_series.polynomial.newton_edge_characteristic_roots.compute"
    )
    assert operation is not None
    example = operation.examples[0]
    result = invoke_operation(operation.operation_id, example.input, Catalog.open())
    assert result.output["characteristic"]["characteristic_polynomial"]["polynomial"]
    roots = result.output["roots"]
    assert len(roots) == 2
    assert all(root["value"]["polynomial"] == ["1", "0", "-2"] for root in roots)
