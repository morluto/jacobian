"""Minimum path decomposition kernel using exhaustive search."""

from __future__ import annotations

from typing import NoReturn

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.path_decomposition._models import (
    MAX_VERTICES,
    PathDecompositionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_minimum_path_decomposition"]

MAX_SEARCH_STATES = 1_000_000
MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("graph",), code=f"path_decomposition.{code}", message=message
    )


def _admit_graph(graph: SimpleUndirectedGraph) -> None:
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject("invalid_graph", "graph must be a simple undirected graph")
    vertex_count = len(graph.vertices)
    if vertex_count > MAX_VERTICES:
        _reject("too_many_vertices", f"at most {MAX_VERTICES} vertices are supported")
    edge_count = len(graph.edges)
    adjacency: dict[str, set[str]] = {v: set() for v in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    if edge_count:
        # Count paths in the actual graph, stopping once the work budget is
        # already impossible. This preserves sparse graphs that the complete
        # graph envelope rejected.
        candidate_bound = max(
            _count_simple_paths(adjacency, MAX_SEARCH_STATES * 2) // 2,
            1,
        )
        search_bound = 1
        for _ in range(edge_count):
            search_bound *= max(candidate_bound, 1)
            if search_bound > MAX_SEARCH_STATES:
                _reject(
                    "search_work_bound",
                    "the exact path-partition search exceeds its bounded work envelope",
                )

    try:
        source_bytes = len(encode_strict_json(graph.model_dump(mode="json")))
        max_label_bytes = max(
            (len(encode_strict_json(vertex)) for vertex in graph.vertices), default=2
        )
    except ValueError as exc:
        _reject("source_representation", str(exc))
    path_bytes = _array_size([max_label_bytes] * vertex_count)
    paths_bytes = _array_size([path_bytes] * edge_count) if edge_count else 2
    result_bytes = strict_json_object_size(
        (
            ("graph", source_bytes),
            ("path_count", 3),
            ("paths", paths_bytes),
        )
    )
    if result_bytes > MAX_RESULT_BYTES:
        _reject(
            "result_size_bound",
            f"the path decomposition result exceeds the {MAX_RESULT_BYTES}-byte output bound",
        )


def _count_simple_paths(
    adjacency: dict[str, set[str]],
    limit: int,
) -> int:
    count = 0

    def visit(current: str, visited: frozenset[str]) -> None:
        nonlocal count
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            count += 1
            if count > limit:
                return
            visit(neighbor, visited | {neighbor})
            if count > limit:
                return

    for start in adjacency:
        visit(start, frozenset({start}))
        if count > limit:
            return limit + 1
    return count


def compute_minimum_path_decomposition(
    graph: SimpleUndirectedGraph,
) -> PathDecompositionResult:
    """Return the exact minimum path number and a realizing partition.

    The path number p(G) is the minimum number of edge-disjoint simple
    paths whose union is E(G). Uses exhaustive search over edge partitions.
    """
    _admit_graph(graph)
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
