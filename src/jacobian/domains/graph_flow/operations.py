"""Domain adapter for graph flow and cut operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

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


def _build_digraph(graph: FlowGraph) -> nx.DiGraph[int]:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for edge in graph.edges:
        g.add_edge(edge.source, edge.target, capacity=edge.capacity.as_fraction())
    return g


def compute_max_flow(request: MaxFlowRequest) -> MaxFlowResult:
    g = _build_digraph(request.graph)
    flow_value = nx.maximum_flow_value(g, request.source, request.sink)
    if not isinstance(flow_value, (int, Fraction)):
        raise RuntimeError("NetworkX did not preserve the exact flow value")
    return MaxFlowResult(
        flow_value=CanonicalRational(
            num=format_canonical_integer(flow_value.numerator),
            den=format_canonical_integer(flow_value.denominator),
        ),
        source=request.source,
        sink=request.sink,
    )


def compute_min_cut(request: MinCutRequest) -> MinCutResult:
    g = _build_digraph(request.graph)
    cut_value, partition = nx.minimum_cut(g, request.source, request.sink)
    if not isinstance(cut_value, (int, Fraction)):
        raise RuntimeError("NetworkX did not preserve the exact cut value")
    reachable, unreachable = partition
    return MinCutResult(
        cut_value=CanonicalRational(
            num=format_canonical_integer(cut_value.numerator),
            den=format_canonical_integer(cut_value.denominator),
        ),
        reachable=tuple(sorted(reachable)),
        unreachable=tuple(sorted(unreachable)),
    )
