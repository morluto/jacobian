"""Domain-owned graph flow and cut operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import networkx as nx

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.flow._models import (
    EdgeDisjointPathsRequest,
    EdgeDisjointPathsResult,
    FlowEdgeResult,
    FlowEdgeValue,
    FlowGraph,
    MaxFlowRequest,
    MaxFlowResult,
    MinCostFlowRequest,
    MinCostFlowResult,
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


def compute_min_cost_flow(request: MinCostFlowRequest) -> MinCostFlowResult:
    """Compute minimum-cost flow with demands using exact integer arithmetic.

    All rational capacities and costs are scaled to integers by a common
    denominator before calling NetworkX's network simplex.  The integer
    results are then divided back to exact rationals.
    """
    edges = request.graph.edges
    capacities = [e.capacity.as_fraction() for e in edges]
    costs = [e.cost.as_fraction() for e in edges]

    # Compute the least common multiple of all denominators so that
    # scaling produces integers.
    from math import lcm

    scale = 1
    for frac in capacities + costs:
        scale = lcm(scale, frac.denominator)

    # Build graph with integer capacities and costs.
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(request.graph.vertex_count))
    for node in range(request.graph.vertex_count):
        g.nodes[node]["demand"] = request.demands[node]
    for edge, cap, cost in zip(edges, capacities, costs, strict=True):
        g.add_edge(
            edge.source,
            edge.target,
            capacity=int(cap * scale),
            weight=int(cost * scale),
        )
    try:
        flow_cost_int, flow_dict = nx.network_simplex(g)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return MinCostFlowResult(
            total_cost=_rational(0),
            feasible=False,
            flow_edges=(),
        )

    flow_edges: list[FlowEdgeResult] = []
    for source_node, targets in flow_dict.items():
        for target_node, flow_amount in targets.items():
            if flow_amount != 0:
                flow_edges.append(
                    FlowEdgeResult(
                        source=source_node,
                        target=target_node,
                        flow=_rational(int(flow_amount)),
                    )
                )
    # The integer flow cost is sum(scaled_weight * flow) = sum((cost * scale) * flow)
    # = scale * sum(cost * flow) = scale * total_cost.
    # Therefore total_cost = flow_cost_int / scale.
    total_cost = Fraction(int(flow_cost_int), scale)
    return MinCostFlowResult(
        total_cost=_rational(total_cost),
        feasible=True,
        flow_edges=tuple(flow_edges),
    )
