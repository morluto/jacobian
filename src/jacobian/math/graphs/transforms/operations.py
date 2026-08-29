"""Exact graph transform kernels backed by NetworkX."""

from __future__ import annotations

from typing import Any

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.transforms._path_profile_models import (
    MAX_PATH_PROFILE_SEARCH_WORK,
    PathProfileResult,
    PathProfileRow,
    _canonical_max_degree,
    _path_prefix_work_bound,
    _path_profile_result_bytes,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "complement",
    "graph_power",
    "induced_subgraph",
    "line_graph",
    "path_profile",
]


def _to_networkx(vertex_count: int, edges: list[tuple[int, int]]) -> Any:
    import networkx as nx

    graph: Any = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    graph.add_edges_from(edges)
    return graph


def _from_networkx(graph: Any) -> tuple[int, list[tuple[int, int]]]:
    return (graph.number_of_nodes(), list(graph.edges()))


def complement(
    vertex_count: int, edges: list[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]]:
    """Return the complement of a simple graph on vertices 0..vertex_count-1."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    return _from_networkx(nx.complement(graph))


def induced_subgraph(
    vertex_count: int, edges: list[tuple[int, int]], vertices: list[int]
) -> tuple[int, list[tuple[int, int]]]:
    """Return the induced subgraph on ``vertices``, reindexed 0..len-1."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    subgraph = nx.induced_subgraph(graph, vertices)
    old_to_new = {old: new for new, old in enumerate(vertices)}
    result_edges = [
        (old_to_new[source], old_to_new[target]) for source, target in subgraph.edges()
    ]
    return (len(vertices), result_edges)


def line_graph(
    vertex_count: int, edges: list[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]]:
    """Return the line graph with vertices reindexed 0..|E(G)|-1."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    transformed = nx.line_graph(graph)
    node_to_index = {node: index for index, node in enumerate(transformed.nodes())}
    result_edges = [
        (node_to_index[source], node_to_index[target])
        for source, target in transformed.edges()
    ]
    return (len(transformed.nodes()), result_edges)


def graph_power(
    vertex_count: int, edges: list[tuple[int, int]], power: int
) -> tuple[int, list[tuple[int, int]]]:
    """Return the ``power``-th power of a simple graph."""

    import networkx as nx

    graph = _to_networkx(vertex_count, edges)
    return _from_networkx(nx.power(graph, power))


def _admit_path_profile(graph: SimpleUndirectedGraph, path_length: int) -> None:
    vertex_count = len(graph.vertices)
    degree_bound = _canonical_max_degree(graph)
    work = vertex_count * _path_prefix_work_bound(
        vertex_count, degree_bound, path_length
    )
    if work > MAX_PATH_PROFILE_SEARCH_WORK:
        raise OperationDomainValidationError(
            location=("graph", "path_length"),
            code="graph.path_profile_search_exceeds_work_budget",
            message=(
                "fixed-length simple path profile search exceeds the "
                f"{MAX_PATH_PROFILE_SEARCH_WORK}-node work budget"
            ),
        )
    output_limit = CanonicalLimits().max_output_bytes
    if _path_profile_result_bytes(graph, path_length) > output_limit:
        raise OperationDomainValidationError(
            location=("graph", "path_length"),
            code="graph.path_profile_result_exceeds_output_budget",
            message=(
                "fixed-length simple path profile result exceeds the canonical "
                f"{output_limit}-byte output budget"
            ),
        )


def path_profile(graph: SimpleUndirectedGraph, path_length: int) -> PathProfileResult:
    """Count simple paths of a fixed length for every ordered endpoint pair."""

    _admit_path_profile(graph, path_length)
    vertices = list(graph.vertices)
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    rows: list[PathProfileRow] = []
    for source in vertices:
        counts = _count_paths_by_endpoint(source, path_length, adjacency)
        rows.extend(
            PathProfileRow(source=source, target=target, path_count=counts[target])
            for target in vertices
            if target in counts
        )
    return PathProfileResult(source=graph, path_length=path_length, rows=rows)


def _count_paths_by_endpoint(
    source: str,
    length: int,
    adjacency: dict[str, set[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(current: str, steps_left: int, visited: set[str]) -> None:
        if steps_left == 0:
            counts[current] = counts.get(current, 0) + 1
            return
        next_visited = visited | {current}
        for neighbor in adjacency[current]:
            if neighbor not in next_visited:
                visit(neighbor, steps_left - 1, next_visited)

    visit(source, length, set())
    return counts
