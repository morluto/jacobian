"""Domain-owned graph flow and cut operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import networkx as nx

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.flows._models import (
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
    _bounded_denominator_scale,
)


def _admit_terminals(
    request: MaxFlowRequest | MinCutRequest | EdgeDisjointPathsRequest,
) -> None:
    if not 0 <= request.source < request.graph.vertex_count:
        raise OperationDomainValidationError(
            location=("source",),
            code="graph.source_must_be_in_0_graph_vertex_count_1",
            message="source must be in 0..graph.vertex_count-1",
        )
    if not 0 <= request.sink < request.graph.vertex_count:
        raise OperationDomainValidationError(
            location=("sink",),
            code="graph.sink_must_be_in_0_graph_vertex_count_1",
            message="sink must be in 0..graph.vertex_count-1",
        )
    if request.source == request.sink:
        raise OperationDomainValidationError(
            location=("source", "sink"),
            code="graph.source_and_sink_must_be_distinct",
            message="source and sink must be distinct",
        )


def _admit_min_cost_flow(request: MinCostFlowRequest) -> None:
    if len(request.demands) != request.graph.vertex_count:
        raise OperationDomainValidationError(
            location=("demands",),
            code="graph.demands_length_must_match_vertex_count",
            message="demands length must match vertex_count",
        )
    if sum(request.demands) != 0:
        raise OperationDomainValidationError(
            location=("demands",),
            code="graph.demands_must_sum_to_zero",
            message="demands must sum to zero",
        )
    try:
        _bounded_denominator_scale(
            tuple(edge.capacity.as_integer_ratio()[1] for edge in request.graph.edges),
            "capacity",
        )
        _bounded_denominator_scale(
            tuple(edge.cost.as_integer_ratio()[1] for edge in request.graph.edges),
            "cost",
        )
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.flow.derived_scale_bound",
            message=str(exc),
        ) from exc


def _build_digraph(graph: FlowGraph) -> nx.DiGraph[int]:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for edge in graph.edges:
        g.add_edge(edge.source, edge.target, capacity=edge.capacity.as_fraction())
    return g


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    return CanonicalRational.from_fraction(Fraction(value))


def compute_max_flow(request: MaxFlowRequest) -> MaxFlowResult:
    _admit_terminals(request)
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
    _admit_terminals(request)
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
    _admit_terminals(request)
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

    Rational capacities, demands, and costs are scaled to integers by two
    derived scales before calling NetworkX's network simplex: the flow scale
    ``F`` (the LCM of the capacity denominators) makes every capacity and
    demand an exact integer, and the cost scale ``C`` (the LCM of the cost
    denominators) makes every per-unit cost an exact integer.  A backend flow
    of ``f'`` is therefore ``f'/F`` units of source flow and a backend
    objective of ``o'`` satisfies ``o' = F * C * total_cost``, so both are
    divided back exactly.  The request model bounds the derived scales before
    any backend graph is built.
    """
    _admit_min_cost_flow(request)
    edges = request.graph.edges

    flow_scale = _bounded_denominator_scale(
        tuple(edge.capacity.as_integer_ratio()[1] for edge in edges), "capacity"
    )
    cost_scale = _bounded_denominator_scale(
        tuple(edge.cost.as_integer_ratio()[1] for edge in edges), "cost"
    )

    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(request.graph.vertex_count))
    for node in range(request.graph.vertex_count):
        g.nodes[node]["demand"] = request.demands[node] * flow_scale
    for edge in edges:
        capacity_num, capacity_den = edge.capacity.as_integer_ratio()
        cost_num, cost_den = edge.cost.as_integer_ratio()
        g.add_edge(
            edge.source,
            edge.target,
            capacity=capacity_num * (flow_scale // capacity_den),
            weight=cost_num * (cost_scale // cost_den),
        )
    try:
        flow_cost_int, flow_dict = nx.network_simplex(g)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return MinCostFlowResult._from_kernel(
            request,
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
                        flow=_rational(Fraction(int(flow_amount), flow_scale)),
                    )
                )
    # A backend objective is sum(cost * cost_scale * flow * flow_scale)
    # over the returned source-unit flows, so the exact public objective
    # divides it by both scales.
    total_cost = Fraction(int(flow_cost_int), flow_scale * cost_scale)
    return MinCostFlowResult._from_kernel(
        request,
        total_cost=_rational(total_cost),
        feasible=True,
        flow_edges=tuple(flow_edges),
    )
