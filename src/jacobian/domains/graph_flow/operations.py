"""Domain adapter for graph flow and cut operations."""

from __future__ import annotations

from fractions import Fraction

import networkx as nx

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.graph_flow import (
    FlowGraph,
    MaxFlowRequest,
    MaxFlowResult,
    MinCutRequest,
    MinCutResult,
)


def _build_digraph(graph: FlowGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for edge in graph.edges:
        cap = edge.capacity.as_fraction()
        g.add_edge(edge.source, edge.target, capacity=float(cap))
    return g


def compute_max_flow(request: MaxFlowRequest) -> MaxFlowResult:
    if request.source == request.sink:
        raise ValueError("source and sink must be distinct")
    g = _build_digraph(request.graph)
    flow_value = nx.maximum_flow_value(g, request.source, request.sink)
    frac = Fraction(flow_value).limit_denominator(10**18)
    return MaxFlowResult(
        flow_value=CanonicalRational(
            num=format_canonical_integer(frac.numerator),
            den=format_canonical_integer(frac.denominator),
        ),
        source=request.source,
        sink=request.sink,
    )


def compute_min_cut(request: MinCutRequest) -> MinCutResult:
    if request.source == request.sink:
        raise ValueError("source and sink must be distinct")
    g = _build_digraph(request.graph)
    cut_value, partition = nx.minimum_cut(g, request.source, request.sink)
    reachable, unreachable = partition
    frac = Fraction(cut_value).limit_denominator(10**18)
    return MinCutResult(
        cut_value=CanonicalRational(
            num=format_canonical_integer(frac.numerator),
            den=format_canonical_integer(frac.denominator),
        ),
        reachable=tuple(sorted(reachable)),
        unreachable=tuple(sorted(unreachable)),
    )
