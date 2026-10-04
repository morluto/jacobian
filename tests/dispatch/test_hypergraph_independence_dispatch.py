"""Public admission boundary for carrier-sized hypergraph independence input.

The advertised example for ``hypergraph.independence_number.compute`` uses
three vertices, and the math lane exercises the native kernel directly. Only
a request that travels the public parsing and result-delivery path at full
carrier scale pins the accepted boundary: a regression that lowered a limit
during wire parsing or envelope encoding would leave both other tests green
while callers could no longer execute a request that previously worked.
"""

from __future__ import annotations

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_math_run_admits_carrier_sized_edgeless_source() -> None:
    vertices = [f"v{i:03}" for i in reversed(range(256))]
    result = invoke_operation(
        "hypergraph.independence_number.compute",
        {"hypergraph": {"vertices": vertices, "edges": []}},
        Catalog.open(),
    )
    assert result.output["independence_number"] == 256
    assert result.output["incumbent_vertices"] == vertices
    assert result.output["solver_calls"] == 0
