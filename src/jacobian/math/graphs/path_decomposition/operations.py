"""Minimum path decomposition kernel using exhaustive search."""

from __future__ import annotations

from jacobian.math.graphs.path_decomposition._models import (
    PathDecompositionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_minimum_path_decomposition"]


def compute_minimum_path_decomposition(
    graph: SimpleUndirectedGraph,
) -> PathDecompositionResult:
    """Return the exact minimum path number and a realizing partition.

    The path number p(G) is the minimum number of edge-disjoint simple
    paths whose union is E(G). Uses exhaustive search over edge partitions.
    """
    edges = list(graph.edges)
    if not edges:
        return PathDecompositionResult(graph=graph, path_count=0, paths=())

    adjacency: dict[str, set[str]] = {v: set() for v in graph.vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    all_paths = _find_all_simple_paths(adjacency)
    all_paths = [p for p in all_paths if p]

    edge_set = frozenset(edges)
    best = _minimum_cover(edge_set, all_paths)

    if best is None:
        return PathDecompositionResult(graph=graph, path_count=len(edges), paths=())

    path_vertices = []
    for path_edges in best:
        vertices = _path_to_vertices(path_edges)
        if vertices:
            path_vertices.append(tuple(vertices))
    return PathDecompositionResult(
        graph=graph,
        path_count=len(best),
        paths=tuple(path_vertices),
    )


def _find_all_simple_paths(
    adjacency: dict[str, set[str]],
) -> list[frozenset[tuple[str, str]]]:
    """Find all simple paths as sets of edges."""
    paths: set[frozenset[tuple[str, str]]] = set()
    for start in adjacency:
        _dfs_paths(start, frozenset(), frozenset({start}), adjacency, paths)
    paths.add(frozenset())
    return list(paths)


def _dfs_paths(
    current: str,
    edges_used: frozenset[tuple[str, str]],
    vertices_visited: frozenset[str],
    adjacency: dict[str, set[str]],
    paths: set[frozenset[tuple[str, str]]],
) -> None:
    if edges_used:
        paths.add(edges_used)
    for neighbor in sorted(adjacency[current]):
        if neighbor in vertices_visited:
            continue
        edge = (min(current, neighbor), max(current, neighbor))
        if edge in edges_used:
            continue
        _dfs_paths(
            neighbor,
            edges_used | {edge},
            vertices_visited | {neighbor},
            adjacency,
            paths,
        )


def _minimum_cover(
    edge_set: frozenset[tuple[str, str]],
    candidates: list[frozenset[tuple[str, str]]],
) -> list[frozenset[tuple[str, str]]] | None:
    """Find the minimum number of paths that partition all edges."""
    if not edge_set:
        return []
    best: list[frozenset[tuple[str, str]]] | None = None
    for path in candidates:
        if path <= edge_set:
            remaining = edge_set - path
            result = _minimum_cover(remaining, candidates)
            if result is not None and (best is None or len(result) + 1 < len(best)):
                best = [path, *result]
    return best


def _path_to_vertices(path_edges: frozenset[tuple[str, str]]) -> list[str]:
    if not path_edges:
        return []
    adj: dict[str, list[str]] = {}
    for a, b in path_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = None
    for v, neighbors in adj.items():
        if len(neighbors) == 1:
            start = v
            break
    if start is None:
        start = next(iter(adj))
    vertices = [start]
    current = start
    used_edges: set[tuple[str, str]] = set()
    while len(vertices) < len(path_edges) + 1:
        for neighbor in adj[current]:
            edge = (min(current, neighbor), max(current, neighbor))
            if edge not in used_edges:
                used_edges.add(edge)
                vertices.append(neighbor)
                current = neighbor
                break
    return vertices
