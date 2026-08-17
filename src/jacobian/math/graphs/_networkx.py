"""Private NetworkX backend for public graph operations."""

from __future__ import annotations

from typing import Any, cast

import networkx as nx

from jacobian.math.graphs.values import GraphCompositionInput, SimpleUndirectedGraph


def simple_graph(graph: nx.Graph[Any]) -> nx.Graph[Any]:
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("graph must be undirected and simple")
    if graph.number_of_nodes() > 32:
        raise ValueError("graph may contain at most 32 vertices")
    if nx.number_of_selfloops(graph):
        raise ValueError("graph must not contain self-loops")
    return graph


def triangle_count(graph: nx.Graph[Any]) -> int:
    counts = cast(dict[Any, int], nx.triangles(simple_graph(graph)))
    return sum(counts.values()) // 3


def diameter(graph: nx.Graph[Any]) -> int:
    value = simple_graph(graph)
    if not value or not nx.is_connected(value):
        raise ValueError("diameter requires a nonempty connected graph")
    return int(nx.diameter(value))


def is_eulerian(graph: nx.Graph[Any]) -> bool:
    g = simple_graph(graph)
    if len(g) == 0:
        return False
    return bool(nx.is_eulerian(g))


def compose_graphs(value: GraphCompositionInput) -> SimpleUndirectedGraph:
    """Apply one composition and return its canonical semantic graph value."""

    left = graph_from_value(value.left)
    right = graph_from_value(value.right) if value.right is not None else None
    if value.operation == "DISJOINT_UNION":
        if right is None:  # guarded by GraphCompositionInput
            raise ValueError("disjoint union requires a right graph")
        result = nx.disjoint_union(left, right)
    elif value.operation == "JOIN":
        if right is None:  # guarded by GraphCompositionInput
            raise ValueError("join requires a right graph")
        result = nx.full_join(left, right, rename=("L", "R"))
    elif value.operation == "COMPLEMENT":
        result = nx.complement(left)
    elif value.operation == "LEXICOGRAPHIC_PRODUCT":
        if right is None:  # guarded by GraphCompositionInput
            raise ValueError("lexicographic product requires a right graph")
        result = nx.lexicographic_product(left, right)
    else:  # pragma: no cover - closed Literal validated by Pydantic
        raise ValueError(f"unsupported composition operation: {value.operation}")
    return graph_value(result)


def graph_from_value(value: SimpleUndirectedGraph) -> nx.Graph[str]:
    """Convert one immutable graph value to a transient NetworkX graph."""

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(value.vertices)
    graph.add_edges_from(value.edges)
    return graph


def graph_from_graph6(encoded: str) -> nx.Graph[Any]:
    """Decode one standard graph6 payload with NetworkX's maintained codec."""

    try:
        payload = encoded.encode("ascii")
        graph = nx.from_graph6_bytes(payload)
    except (UnicodeEncodeError, ValueError, IndexError, nx.NetworkXError) as exc:
        raise ValueError(
            "graph6 payload is malformed or uses an extended header"
        ) from exc
    return graph


def graph6_canonical_bytes(graph: nx.Graph[Any]) -> bytes:
    """Return headerless graph6 bytes without NetworkX's trailing newline."""

    encoded = bytes(nx.to_graph6_bytes(graph, header=False))
    return encoded[:-1] if encoded.endswith(b"\n") else encoded


def graph_value(graph: nx.Graph[Any]) -> SimpleUndirectedGraph:
    """Canonicalize a transient NetworkX graph as an immutable graph value."""

    ordered_nodes = tuple(sorted(graph.nodes))
    labels = {node: f"v{index}" for index, node in enumerate(ordered_nodes)}
    edges = tuple(
        sorted(
            (labels[source], labels[target])
            if labels[source] < labels[target]
            else (labels[target], labels[source])
            for source, target in graph.edges
        )
    )
    return SimpleUndirectedGraph(
        vertices=tuple(labels[node] for node in ordered_nodes),
        edges=edges,
    )
