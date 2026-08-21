"""Domain-owned graph cycle and subgraph-pattern operations."""

from __future__ import annotations

import networkx as nx

from jacobian.math.graphs.cycle_pattern._models import (
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    SubgraphEmbedding,
    SubgraphPatternRequest,
    SubgraphPatternResult,
    UndirectedGraph,
)


def _build_graph(graph: UndirectedGraph) -> nx.Graph[int]:
    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(graph.vertex_count))
    for source, target in graph.edges:
        g.add_edge(source, target)
    return g


def decide_fixed_length_cycle(
    request: FixedLengthCycleRequest,
) -> FixedLengthCycleResult:
    """Decide whether a graph contains a simple cycle of a given length.

    A backtracking DFS rooted at each vertex finds the lexicographically
    smallest cycle of the requested length, if one exists.  To avoid
    duplicates, only cycles whose minimum vertex is the root are returned.
    """
    g = _build_graph(request.graph)
    k = request.length
    n = request.graph.vertex_count

    for start in range(n):
        found = _search_cycle_from(g, start, k)
        if found is not None:
            return FixedLengthCycleResult(
                vertex_count=n,
                length=k,
                exists=True,
                cycle=tuple(found),
            )

    return FixedLengthCycleResult(
        vertex_count=n,
        length=k,
        exists=False,
    )


def _search_cycle_from(
    g: nx.Graph[int],
    start: int,
    k: int,
) -> list[int] | None:
    """Find the lexicographically smallest simple k-cycle through ``start``.

    Builds the path ``[start, v_1, ..., v_{k-1}]`` where each ``v_i > start``
    and checks that ``v_{k-1}`` is adjacent to ``start``, closing the cycle.
    Enforcing all intermediate vertices to exceed ``start`` avoids returning
    duplicate cycles.
    """
    path: list[int] = [start]
    visited: set[int] = {start}

    def dfs(depth: int) -> list[int] | None:
        if depth == k:
            if start in g.neighbors(path[-1]):
                return list(path)
            return None
        current = path[-1]
        for neighbor in sorted(g.neighbors(current)):
            if neighbor in visited:
                continue
            if neighbor <= start:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            result = dfs(depth + 1)
            if result is not None:
                return result
            path.pop()
            visited.discard(neighbor)
        return None

    return dfs(1)


def find_subgraph_pattern(
    request: SubgraphPatternRequest,
) -> SubgraphPatternResult:
    """Find an injective subgraph embedding of a pattern into a host graph.

    Backtracking search assigns pattern vertices to distinct host vertices,
    checking edge preservation at each step.  Returns the lexicographically
    smallest embedding, if one exists.
    """
    host_g = _build_graph(request.host)
    pattern_g = _build_graph(request.pattern)
    pattern_vertices = sorted(pattern_g.nodes())
    n_pattern = len(pattern_vertices)

    pattern_adj: dict[int, set[int]] = {
        v: set(pattern_g.neighbors(v)) for v in pattern_vertices
    }
    host_adj: dict[int, set[int]] = {
        v: set(host_g.neighbors(v)) for v in host_g.nodes()
    }

    mapping: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == n_pattern:
            return True
        pv = pattern_vertices[idx]
        for hv in sorted(host_g.nodes()):
            if hv in used:
                continue
            ok = True
            for prev_pv in pattern_vertices[:idx]:
                if prev_pv not in pattern_adj[pv]:
                    continue
                if mapping[prev_pv] not in host_adj.get(hv, set()):
                    ok = False
                    break
            if not ok:
                continue
            mapping[pv] = hv
            used.add(hv)
            if backtrack(idx + 1):
                return True
            used.discard(hv)
            del mapping[pv]
        return False

    if backtrack(0):
        emb = SubgraphEmbedding(
            mapping=tuple((pv, mapping[pv]) for pv in pattern_vertices),
        )
        return SubgraphPatternResult(
            host_vertex_count=request.host.vertex_count,
            pattern_vertex_count=request.pattern.vertex_count,
            exists=True,
            embedding=emb,
        )

    return SubgraphPatternResult(
        host_vertex_count=request.host.vertex_count,
        pattern_vertex_count=request.pattern.vertex_count,
        exists=False,
    )


__all__ = [
    "decide_fixed_length_cycle",
    "find_subgraph_pattern",
]
