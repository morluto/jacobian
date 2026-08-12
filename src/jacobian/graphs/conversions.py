"""Explicit conversions between graph values and graph wire contracts."""

from __future__ import annotations

from jacobian.contracts.graph_isomorphism import (
    SimpleUndirectedGraph as SimpleUndirectedGraphContract,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def graph_value_from_contract(
    graph: SimpleUndirectedGraphContract,
) -> SimpleUndirectedGraph:
    """Convert a validated graph wire contract to its semantic value."""

    return SimpleUndirectedGraph(
        graph_schema_version=graph.graph_schema_version,
        vertices=graph.vertices,
        edges=graph.edges,
    )


def graph_contract_from_value(
    graph: SimpleUndirectedGraph,
) -> SimpleUndirectedGraphContract:
    """Convert a semantic graph value to the graph wire contract."""

    return SimpleUndirectedGraphContract(
        graph_schema_version=graph.graph_schema_version,
        vertices=graph.vertices,
        edges=graph.edges,
    )


__all__ = [
    "graph_contract_from_value",
    "graph_value_from_contract",
]
