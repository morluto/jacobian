"""Minimum path decomposition kernel using exhaustive search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import NoReturn

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.path_decomposition._models import (
    MAX_VERTICES,
    PathDecompositionResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_minimum_path_decomposition"]

MAX_SEARCH_STATES = 1_000_000
MAX_CANDIDATE_EDGE_INCIDENCES = 1_000_000


@dataclass(frozen=True, slots=True)
class _PathSearchPlan:
    candidates: tuple[frozenset[tuple[str, str]], ...]
    candidate_incidences: int
    candidate_checks_bound: int


def _array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("graph",), code=f"path_decomposition.{code}", message=message
    )


def _admit_graph(graph: SimpleUndirectedGraph) -> _PathSearchPlan:
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject("invalid_graph", "graph must be a simple undirected graph")
    vertex_count = len(graph.vertices)
    if vertex_count > MAX_VERTICES:
        _reject("too_many_vertices", f"at most {MAX_VERTICES} vertices are supported")
    adjacency: dict[str, set[str]] = {v: set() for v in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    candidate_sets, candidate_incidences = _find_all_simple_paths(
        adjacency, candidate_limit=MAX_SEARCH_STATES
    )
    candidates = tuple(
        sorted(
            candidate_sets,
            key=lambda path: (-len(path), tuple(sorted(path))),
        )
    )
    candidate_checks_bound = MAX_SEARCH_STATES
    return _PathSearchPlan(
        candidates=candidates,
        candidate_incidences=candidate_incidences,
        candidate_checks_bound=candidate_checks_bound,
    )


def compute_minimum_path_decomposition(
    graph: SimpleUndirectedGraph,
) -> PathDecompositionResult:
    """Return the exact minimum path number and a realizing partition.

    The path number p(G) is the minimum number of edge-disjoint simple
    paths whose union is E(G). Uses exhaustive search over edge partitions.
    """
    plan = _admit_graph(graph)
    edges = list(graph.edges)
    if not edges:
        return PathDecompositionResult(graph=graph, path_count=0, paths=())

    edge_set = frozenset(edges)
    best = _minimum_cover(
        edge_set,
        plan.candidates,
        candidate_incidences=plan.candidate_incidences,
        candidate_checks_bound=plan.candidate_checks_bound,
    )

    if best is None:
        raise AssertionError("single-edge paths must realize every admitted graph")

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
    *,
    candidate_limit: int,
) -> tuple[set[frozenset[tuple[str, str]]], int]:
    """Find all simple paths as sets of edges."""
    paths: set[frozenset[tuple[str, str]]] = set()
    enumeration_steps = [0]
    candidate_incidences = [0]
    for start in adjacency:
        _dfs_paths(
            start,
            frozenset(),
            frozenset({start}),
            adjacency,
            paths,
            candidate_limit=candidate_limit,
            enumeration_steps=enumeration_steps,
            candidate_incidences=candidate_incidences,
        )
    return paths, candidate_incidences[0]


def _dfs_paths(
    current: str,
    edges_used: frozenset[tuple[str, str]],
    vertices_visited: frozenset[str],
    adjacency: dict[str, set[str]],
    paths: set[frozenset[tuple[str, str]]],
    *,
    candidate_limit: int,
    enumeration_steps: list[int],
    candidate_incidences: list[int],
) -> None:
    enumeration_steps[0] += 1
    if enumeration_steps[0] > MAX_SEARCH_STATES:
        _reject(
            "path_enumeration_bound",
            "simple-path enumeration exceeds its bounded work envelope",
        )
    if edges_used:
        if edges_used not in paths:
            candidate_incidences[0] += len(edges_used)
            if candidate_incidences[0] > MAX_CANDIDATE_EDGE_INCIDENCES:
                _reject(
                    "candidate_materialization_bound",
                    "simple-path candidate materialization exceeds its bounded incidence envelope",
                )
            paths.add(edges_used)
        if len(paths) > candidate_limit:
            _reject(
                "search_work_bound",
                "the exact memoized residual-edge search exceeds its bounded work envelope",
            )
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
            candidate_limit=candidate_limit,
            enumeration_steps=enumeration_steps,
            candidate_incidences=candidate_incidences,
        )


def _minimum_cover(
    edge_set: frozenset[tuple[str, str]],
    candidates: tuple[frozenset[tuple[str, str]], ...],
    *,
    candidate_incidences: int,
    candidate_checks_bound: int,
) -> list[frozenset[tuple[str, str]]] | None:
    """Find the minimum partition while solving each residual edge set once."""

    by_edge_lists: dict[tuple[str, str], list[frozenset[tuple[str, str]]]] = {
        edge: [] for edge in edge_set
    }
    for path in candidates:
        for edge in path:
            candidate_incidences += 1
            if candidate_incidences > MAX_CANDIDATE_EDGE_INCIDENCES:
                _reject(
                    "candidate_materialization_bound",
                    "simple-path candidate indexing exceeds its bounded incidence envelope",
                )
            by_edge_lists[edge].append(path)
    by_edge = {edge: tuple(paths) for edge, paths in by_edge_lists.items()}
    candidate_checks = 0

    @cache
    def solve(
        remaining: frozenset[tuple[str, str]],
    ) -> tuple[frozenset[tuple[str, str]], ...] | None:
        nonlocal candidate_checks
        if not remaining:
            return ()
        pivot = min(remaining)
        best: tuple[frozenset[tuple[str, str]], ...] | None = None
        for path in by_edge[pivot]:
            candidate_checks += 1
            if candidate_checks > candidate_checks_bound:
                _reject(
                    "search_work_bound",
                    "the exact memoized residual-edge search exceeds its bounded work envelope",
                )
            if not path <= remaining:
                continue
            suffix = solve(remaining - path)
            if suffix is not None and (best is None or len(suffix) + 1 < len(best)):
                best = (path, *suffix)
        return best

    result = solve(edge_set)
    return list(result) if result is not None else None


def _path_to_vertices(path_edges: frozenset[tuple[str, str]]) -> list[str]:
    if not path_edges:
        return []
    adj: dict[str, list[str]] = {}
    for a, b in sorted(path_edges):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = min(vertex for vertex, neighbors in adj.items() if len(neighbors) == 1)
    vertices = [start]
    current = start
    used_edges: set[tuple[str, str]] = set()
    while len(vertices) < len(path_edges) + 1:
        for neighbor in sorted(adj[current]):
            edge = (min(current, neighbor), max(current, neighbor))
            if edge not in used_edges:
                used_edges.add(edge)
                vertices.append(neighbor)
                current = neighbor
                break
    return vertices
