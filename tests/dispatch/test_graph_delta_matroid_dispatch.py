"""Catalog and dispatch boundaries for looped-graph delta-matroids.

The mathematical contract of the conversion is owned by
``tests/math/graphs/delta_matroids``. Booting the complete product
boundary is forbidden there, so the published catalog and dispatch
surfaces are exercised from this owner instead.
"""

from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_global_catalog_discovers_and_executes_graph_conversion() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("graph.looped_adjacency_delta_matroid.compute")
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["matrix"]["entries"] == [[1, 1], [1, 0]]
    assert result.output["delta_matroid"]["feasible"] == [[], [0], [0, 1]]
