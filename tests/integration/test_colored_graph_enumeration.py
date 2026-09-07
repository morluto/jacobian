"""Colored-family native and wire paths share exact enumeration results."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.enumeration import (
    ColorPairCost,
    ConnectedColoredGraphFamily,
    connected_colored_graphs,
)
from jacobian.math.graphs.enumeration._models import ConnectedColoredGraphsRequest


def test_native_parameters_preserve_wire_result() -> None:
    request = ConnectedColoredGraphsRequest(
        palette=("outside", "root"),
        edge_costs=(
            ColorPairCost(colors=("outside", "outside"), cost=2),
            ColorPairCost(colors=("outside", "root"), cost=1),
        ),
        vertex_bound=4,
        cost_bound=3,
    )
    native = connected_colored_graphs(
        request.palette,
        request.edge_costs,
        vertex_bound=request.vertex_bound,
        cost_bound=request.cost_bound,
    )
    wire = invoke_operation(
        "graph.colored.connected_family.enumerate",
        request.model_dump(mode="json"),
        Catalog.open(),
    ).output
    assert ConnectedColoredGraphFamily.model_validate(wire) == native
