"""Domain adapter for graph flow and cut operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import networkx as nx

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.graph_flow import (
    EdgeDisjointPathsRequest,
    EdgeDisjointPathsResult,
    FlowEdgeValue,
    FlowGraph,
    MaxFlowRequest,
    MaxFlowResult,
    MinCutRequest,
    MinCutResult,
)


def _build_digraph(graph: FlowGraph) -> nx.DiGraph[int]:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for edge in graph.edges:
        g.add_edge(edge.source, edge.target, capacity=edge.capacity.as_fraction())
    return g


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    frac = Fraction(value)
    return CanonicalRational(
        num=format_canonical_integer(frac.numerator),
        den=format_canonical_integer(frac.denominator),
    )


def compute_max_flow(request: MaxFlowRequest) -> MaxFlowResult:
    g = _build_digraph(request.graph)
    flow_value, flow_dict = nx.maximum_flow(g, request.source, request.sink)
    if not isinstance(flow_value, (int, Fraction)):
        raise RuntimeError("NetworkX did not preserve the exact flow value")

    # Build a per-edge flow decomposition so the caller can independently
    # verify conservation and capacity constraints.
    flow_edges: list[FlowEdgeValue] = []
    for source_node, targets in flow_dict.items():
        for target_node, flow_amount in targets.items():
            if flow_amount != 0:
                flow_edges.append(
                    FlowEdgeValue(
                        source=source_node,
                        target=target_node,
                        flow=_rational(flow_amount),
                    )
                )
    return MaxFlowResult(
        flow_value=_rational(flow_value),
        source=request.source,
        sink=request.sink,
        flow_edges=tuple(flow_edges),
    )


def compute_min_cut(request: MinCutRequest) -> MinCutResult:
    g = _build_digraph(request.graph)
    cut_value, partition = nx.minimum_cut(g, request.source, request.sink)
    if not isinstance(cut_value, (int, Fraction)):
        raise RuntimeError("NetworkX did not preserve the exact cut value")
    reachable, unreachable = partition
    return MinCutResult(
        cut_value=_rational(cut_value),
        reachable=tuple(sorted(reachable)),
        unreachable=tuple(sorted(unreachable)),
    )


def compute_edge_disjoint_paths(
    request: EdgeDisjointPathsRequest,
) -> EdgeDisjointPathsResult:
    """Compute the maximum number of edge-disjoint paths and the explicit paths.

    Uses NetworkX's ``edge_disjoint_paths`` (which internally computes a
    maximum flow with unit capacities and extracts the paths).
    """
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(request.graph.vertex_count))
    for source, target in request.graph.edges:
        g.add_edge(source, target)

    try:
        paths = list(nx.edge_disjoint_paths(g, request.source, request.sink))
    except nx.NetworkXNoPath:
        return EdgeDisjointPathsResult(
            path_count=0,
            paths=(),
            source=request.source,
            sink=request.sink,
        )

    return EdgeDisjointPathsResult(
        path_count=len(paths),
        paths=tuple(tuple(path) for path in paths),
        source=request.source,
        sink=request.sink,
    )
