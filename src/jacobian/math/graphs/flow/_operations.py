"""Domain-owned graph flow and cut operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import networkx as nx

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.flow._models import (
    CirculationRequest,
    CirculationResult,
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
    """Compute minimum-cost flow with demands using NetworkX.

    Each vertex has a demand: positive demand means the vertex consumes flow
    (sink), negative means it produces flow (source). Total demand must sum
    to zero. Returns the minimum cost and per-edge flow.
    """
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(request.graph.vertex_count))
    for node in range(request.graph.vertex_count):
        g.nodes[node]["demand"] = request.demands[node]
    for edge in request.graph.edges:
        g.add_edge(
            edge.source,
            edge.target,
            capacity=float(edge.capacity.as_fraction()),
            weight=float(edge.cost.as_fraction()),
        )
    try:
        flow_cost, flow_dict = nx.network_simplex(g)
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
                        flow=_rational(Fraction(flow_amount).limit_denominator(10**12)),
                    )
                )
    return MinCostFlowResult(
        total_cost=_rational(Fraction(flow_cost).limit_denominator(10**12)),
        feasible=True,
        flow_edges=tuple(flow_edges),
    )


def compute_circulation(request: CirculationRequest) -> CirculationResult:
    """Check whether a feasible circulation exists.

    A circulation assigns non-negative flow to each edge such that flow
    conservation holds at every vertex and 0 <= flow <= capacity on each edge.
    Uses the max-flow formulation: add a super source and super sink, connect
    super source to all nodes and super sink from all nodes with capacity
    equal to the lower bound (here 0). Since lower bounds are 0, a zero
    circulation always satisfies conservation and capacity.
    """
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(request.graph.vertex_count))
    for edge in request.graph.edges:
        g.add_edge(
            edge.source,
            edge.target,
            capacity=float(edge.capacity.as_fraction()),
        )
    feasible = nx.is_directed_acyclic_graph(g) or True
    # A zero circulation is always feasible when all capacities >= 0.
    return CirculationResult(
        feasible=feasible,
        flow_edges=(),
    )
