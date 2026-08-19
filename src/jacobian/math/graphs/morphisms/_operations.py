"""Domain functions for graph morphism operations."""

from __future__ import annotations

from jacobian.math.graphs.morphisms._models import (
    CoreCheckRequest,
    CoreCheckResult,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    HomomorphismFindRequest,
    HomomorphismFindResult,
    RetractionCheckRequest,
    RetractionCheckResult,
)


def _adjacency(graph_edges: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    """Return a set of all directed edges (both directions)."""
    adj: set[tuple[int, int]] = set()
    for u, v in graph_edges:
        adj.add((u, v))
        adj.add((v, u))
    return adj


def _is_homomorphism(
    source_edges: tuple[tuple[int, int], ...],
    target_edges: tuple[tuple[int, int], ...],
    vertex_map: list[int],
) -> bool:
    target_adj = _adjacency(target_edges)
    return all((vertex_map[u], vertex_map[v]) in target_adj for u, v in source_edges)


def compute_homomorphism_check(
    request: HomomorphismCheckRequest,
) -> HomomorphismCheckResult:
    is_h = _is_homomorphism(
        request.source_graph.edges,
        request.target_graph.edges,
        list(request.vertex_map),
    )
    return HomomorphismCheckResult(is_homomorphism=is_h)


def compute_homomorphism_find(
    request: HomomorphismFindRequest,
) -> HomomorphismFindResult:
    source = request.source_graph
    target = request.target_graph
    target_adj = _adjacency(target.edges)

    vertex_map: list[int] = [-1] * source.vertex_count

    def backtrack(pos: int) -> bool:
        if pos == source.vertex_count:
            return True
        for candidate in range(target.vertex_count):
            vertex_map[pos] = candidate
            ok = True
            for u, v in source.edges:
                if (
                    u == pos
                    and vertex_map[v] != -1
                    and (vertex_map[u], vertex_map[v]) not in target_adj
                ):
                    ok = False
                    break
                if (
                    v == pos
                    and vertex_map[u] != -1
                    and (vertex_map[u], vertex_map[v]) not in target_adj
                ):
                    ok = False
                    break
            if ok and backtrack(pos + 1):
                return True
            vertex_map[pos] = -1
        return False

    found = backtrack(0)
    return HomomorphismFindResult(
        found=found,
        vertex_map=tuple(vertex_map) if found else (),
    )


def _is_endomorphism(
    source_edges: tuple[tuple[int, int], ...],
    source_adj: set[tuple[int, int]],
    mapping: list[int],
) -> bool:
    return all((mapping[u], mapping[v]) in source_adj for u, v in source_edges)


def compute_core_check(request: CoreCheckRequest) -> CoreCheckResult:
    """A graph is a core iff it has no non-injective endomorphism."""
    source = request.graph
    source_adj = _adjacency(source.edges)
    vertex_map: list[int] = [-1] * source.vertex_count
    has_non_injective = [False]

    def search_non_injective(pos: int) -> bool:
        if pos == source.vertex_count:
            used = set(vertex_map)
            if len(used) < source.vertex_count:
                has_non_injective[0] = True
                return True
            return False
        for candidate in range(source.vertex_count):
            vertex_map[pos] = candidate
            if _is_endomorphism(
                source.edges, source_adj, vertex_map
            ) and search_non_injective(pos + 1):
                return True
            vertex_map[pos] = -1
        return False

    found = search_non_injective(0)
    return CoreCheckResult(is_core=not found)


def compute_retraction_check(
    request: RetractionCheckRequest,
) -> RetractionCheckResult:
    """Check if a retraction onto an induced subgraph exists."""
    source = request.graph
    subgraph = set(request.subgraph_vertices)

    target_edges = [(u, v) for u, v in source.edges if u in subgraph and v in subgraph]
    target_adj = _adjacency(tuple(target_edges))

    vertex_map: list[int] = [-1] * source.vertex_count
    subgraph_list = list(subgraph)

    for v in subgraph_list:
        vertex_map[v] = v

    remaining = [i for i in range(source.vertex_count) if i not in subgraph]

    def backtrack(pos: int) -> bool:
        if pos == len(remaining):
            return True
        v = remaining[pos]
        for candidate in subgraph_list:
            vertex_map[v] = candidate
            ok = True
            for u, w in source.edges:
                if (
                    u == v
                    and vertex_map[w] != -1
                    and (candidate, vertex_map[w]) not in target_adj
                ):
                    ok = False
                    break
                if (
                    w == v
                    and vertex_map[u] != -1
                    and (vertex_map[u], candidate) not in target_adj
                ):
                    ok = False
                    break
            if ok and backtrack(pos + 1):
                return True
            vertex_map[v] = -1
        return False

    found = backtrack(0)
    return RetractionCheckResult(is_retraction=found)
