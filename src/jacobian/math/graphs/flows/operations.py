"""Native exact operations for directed graph flows."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import networkx as nx

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.flows._models import (
    CostedFlowGraph,
    EdgeDisjointPathsGraph,
    FlowEdgeResult,
    FlowEdgeValue,
    FlowGraph,
    _bounded_denominator_scale,
)

__all__ = [
    "edge_disjoint_paths",
    "max_flow",
    "min_cost_flow",
    "min_cut",
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
