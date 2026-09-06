"""Native exact operations for directed graph flows."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Any

import networkx as nx

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.flows._models import (
    CostedFlowGraph,
    EdgeDisjointPathsGraph,
    EdgeDisjointPathsResult,
    FlowEdgeResult,
    FlowEdgeValue,
    FlowGraph,
    MaxFlowResult,
    MinCutResult,
    _bounded_denominator_scale,
)

__all__ = [
    "edge_disjoint_paths",
    "max_flow",
    "min_cost_flow",
    "min_cut",
    "verify_edge_disjoint_paths",
    "verify_max_flow",
    "verify_min_cut",
]


def _admit_terminals(
    graph: FlowGraph | EdgeDisjointPathsGraph, source: int, sink: int
) -> None:
    if not 0 <= source < graph.vertex_count:
        raise OperationDomainValidationError(
            location=("source",),
            code="graph.source_must_be_in_0_graph_vertex_count_1",
            message="source must be in 0..graph.vertex_count-1",
        )
    if not 0 <= sink < graph.vertex_count:
        raise OperationDomainValidationError(
            location=("sink",),
            code="graph.sink_must_be_in_0_graph_vertex_count_1",
            message="sink must be in 0..graph.vertex_count-1",
        )
    if source == sink:
        raise OperationDomainValidationError(
            location=("source", "sink"),
            code="graph.source_and_sink_must_be_distinct",
            message="source and sink must be distinct",
        )


def _admit_min_cost_flow(graph: CostedFlowGraph, demands: tuple[int, ...]) -> None:
    if len(demands) != graph.vertex_count:
        raise OperationDomainValidationError(
            location=("demands",),
            code="graph.demands_length_must_match_vertex_count",
            message="demands length must match vertex_count",
        )
    if sum(demands) != 0:
        raise OperationDomainValidationError(
            location=("demands",),
            code="graph.demands_must_sum_to_zero",
            message="demands must sum to zero",
        )
    try:
        _bounded_denominator_scale(
            tuple(edge.capacity.as_integer_ratio()[1] for edge in graph.edges),
            "capacity",
        )
        _bounded_denominator_scale(
            tuple(edge.cost.as_integer_ratio()[1] for edge in graph.edges), "cost"
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.flow.derived_scale_bound",
            message=str(exc),
        ) from exc


def _build_digraph(graph: FlowGraph) -> nx.DiGraph[int]:
    result: nx.DiGraph[Any] = nx.DiGraph()
    result.add_nodes_from(range(graph.vertex_count))
    for edge in graph.edges:
        result.add_edge(edge.source, edge.target, capacity=edge.capacity.as_fraction())
    return result


def _rational(value: Fraction | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def max_flow(
    graph: FlowGraph, source: int, sink: int
) -> tuple[CanonicalRational, tuple[FlowEdgeValue, ...]]:
    """Return the exact maximum flow value and its edge decomposition."""

    _admit_terminals(graph, source, sink)
    flow_value, flow_dict = nx.maximum_flow(_build_digraph(graph), source, sink)
    if not isinstance(flow_value, (int, Fraction)):
        raise RuntimeError("NetworkX did not preserve the exact flow value")
    flow_edges = tuple(
        FlowEdgeValue(source=u, target=v, flow=_rational(amount))
        for u, targets in flow_dict.items()
        for v, amount in targets.items()
        if amount != 0
    )
    return _rational(flow_value), flow_edges


def min_cut(
    graph: FlowGraph, source: int, sink: int
) -> tuple[CanonicalRational, tuple[int, ...], tuple[int, ...]]:
    """Return the exact minimum s-t cut value and its partition."""

    _admit_terminals(graph, source, sink)
    cut_value, partition = nx.minimum_cut(_build_digraph(graph), source, sink)
    if not isinstance(cut_value, (int, Fraction)):
        raise RuntimeError("NetworkX did not preserve the exact cut value")
    reachable, unreachable = partition
    return _rational(cut_value), tuple(sorted(reachable)), tuple(sorted(unreachable))


def edge_disjoint_paths(
    graph: EdgeDisjointPathsGraph, source: int, sink: int
) -> tuple[tuple[int, ...], ...]:
    """Return all exact edge-disjoint source-to-sink paths."""

    _admit_terminals(graph, source, sink)
    network: nx.DiGraph[Any] = nx.DiGraph()
    network.add_nodes_from(range(graph.vertex_count))
    network.add_edges_from(graph.edges)
    try:
        return tuple(
            tuple(path) for path in nx.edge_disjoint_paths(network, source, sink)
        )
    except nx.NetworkXNoPath:
        return ()


def min_cost_flow(
    graph: CostedFlowGraph, demands: tuple[int, ...]
) -> tuple[CanonicalRational, bool, tuple[FlowEdgeResult, ...]]:
    """Return the exact minimum-cost flow outcome."""

    _admit_min_cost_flow(graph, demands)
    edges = graph.edges
    flow_scale = _bounded_denominator_scale(
        tuple(edge.capacity.as_integer_ratio()[1] for edge in edges), "capacity"
    )
    cost_scale = _bounded_denominator_scale(
        tuple(edge.cost.as_integer_ratio()[1] for edge in edges), "cost"
    )
    network: nx.DiGraph[Any] = nx.DiGraph()
    network.add_nodes_from(range(graph.vertex_count))
    for node in range(graph.vertex_count):
        network.nodes[node]["demand"] = demands[node] * flow_scale
    for edge in edges:
        capacity_num, capacity_den = edge.capacity.as_integer_ratio()
        cost_num, cost_den = edge.cost.as_integer_ratio()
        network.add_edge(
            edge.source,
            edge.target,
            capacity=capacity_num * (flow_scale // capacity_den),
            weight=cost_num * (cost_scale // cost_den),
        )
    try:
        flow_cost_int, flow_dict = nx.network_simplex(network)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return _rational(0), False, ()
    flow_edges = tuple(
        FlowEdgeResult(
            source=u,
            target=v,
            flow=_rational(Fraction(int(amount), flow_scale)),
        )
        for u, targets in flow_dict.items()
        for v, amount in targets.items()
        if amount != 0
    )
    total_cost = Fraction(int(flow_cost_int), flow_scale * cost_scale)
    return _rational(total_cost), True, flow_edges


def verify_max_flow(claim: MaxFlowResult) -> bool:
    """Check capacities, conservation and absence of residual augmenting paths.

    The network has at most 64 vertices and 512 edges; no solver is rerun.
    """
    _admit_terminals(claim.graph, claim.source, claim.sink)
    flows = {
        (edge.source, edge.target): edge.flow.as_fraction() for edge in claim.flow_edges
    }
    balance = [Fraction() for _ in range(claim.graph.vertex_count)]
    residual: list[set[int]] = [set() for _ in balance]
    for edge in claim.graph.edges:
        u, v = edge.source, edge.target
        amount = flows.get((u, v), Fraction())
        capacity = edge.capacity.as_fraction()
        if not 0 <= amount <= capacity:
            return False
        balance[u] -= amount
        balance[v] += amount
        if amount < capacity:
            residual[u].add(v)
        if amount > 0:
            residual[v].add(u)
    value = claim.flow_value.as_fraction()
    if any(
        amount
        != (-value if vertex == claim.source else value if vertex == claim.sink else 0)
        for vertex, amount in enumerate(balance)
    ):
        return False
    reached = {claim.source}
    frontier = [claim.source]
    while frontier:
        for vertex in residual[frontier.pop()]:
            if vertex not in reached:
                reached.add(vertex)
                frontier.append(vertex)
    return claim.sink not in reached


def verify_min_cut(claim: MinCutResult) -> bool:
    """Check cut capacity and optimality against the admitted source network."""
    _admit_terminals(claim.graph, claim.source, claim.sink)
    left, right = set(claim.reachable), set(claim.unreachable)
    if claim.source not in left or claim.sink not in right:
        return False
    capacity = sum(
        (
            edge.capacity.as_fraction()
            for edge in claim.graph.edges
            if edge.source in left and edge.target in right
        ),
        Fraction(),
    )
    return (
        capacity == claim.cut_value.as_fraction()
        and min_cut(claim.graph, claim.source, claim.sink)[0] == claim.cut_value
    )


def verify_edge_disjoint_paths(claim: EdgeDisjointPathsResult) -> bool:
    """Check path witnesses and the maximum cardinality in the bounded graph."""
    _admit_terminals(claim.graph, claim.source, claim.sink)
    edges = set(claim.graph.edges)
    used: set[tuple[int, int]] = set()
    if claim.path_count != len(claim.paths):
        return False
    for path in claim.paths:
        if (
            path[0] != claim.source
            or path[-1] != claim.sink
            or len(set(path)) != len(path)
        ):
            return False
        for edge in pairwise(path):
            if edge not in edges or edge in used:
                return False
            used.add(edge)
    return claim.path_count == len(
        edge_disjoint_paths(claim.graph, claim.source, claim.sink)
    )
