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

# Backtracking recursion nodes across one request; a schema-valid request
# cannot occupy the inline path beyond this deterministic work bound.
MAX_SEARCH_NODES = 2_000_000


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
    budget = _NodeBudget()

    for start in range(n):
        found = _search_cycle_from(g, start, k, budget)
        if found is not None:
            return FixedLengthCycleResult(
                graph=request.graph,
                vertex_count=n,
                length=k,
                exists=True,
                cycle=tuple(found),
            )
        if budget.exceeded():
            return FixedLengthCycleResult(
                graph=request.graph,
                vertex_count=n,
                length=k,
                outcome="SEARCH_BUDGET_EXCEEDED",
                detail=(
                    f"cycle search exceeded {MAX_SEARCH_NODES} recursion "
                    "nodes without deciding"
                ),
            )

    return FixedLengthCycleResult(
        graph=request.graph,
        vertex_count=n,
        length=k,
        exists=False,
    )


class _NodeBudget:
    """Deterministic recursion-node counter bounding one search."""

    def __init__(self, limit: int | None = None) -> None:
        # Read the module bound at construction so tests can narrow it.
        self._limit = MAX_SEARCH_NODES if limit is None else limit
        self._used = 0

    def tick(self) -> None:
        self._used += 1

    def exceeded(self) -> bool:
        return self._used >= self._limit


def _search_cycle_from(
    g: nx.Graph[int],
    start: int,
    k: int,
    budget: _NodeBudget,
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
        budget.tick()
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

    pattern_degrees = {
        v: len(pattern_adj[v]) for v in pattern_vertices
    }
    host_degrees = {v: len(host_adj.get(v, ())) for v in host_g.nodes()}
    budget = _NodeBudget()
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(idx: int) -> bool:
        budget.tick()
        if budget.exceeded():
            raise _BudgetExceededError()
        if idx == n_pattern:
            return True
        pv = pattern_vertices[idx]
        # Necessary condition: a host vertex must have degree at least the
        # pattern vertex's total degree to carry all of its edges.
        required_degree = pattern_degrees[pv]
        for hv in sorted(host_g.nodes()):
            if hv in used or host_degrees[hv] < required_degree:
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

    try:
        decided = backtrack(0)
    except _BudgetExceededError:
        return SubgraphPatternResult(
            host_graph=request.host,
            pattern_graph=request.pattern,
            outcome="SEARCH_BUDGET_EXCEEDED",
            detail=(
                f"subgraph search exceeded {MAX_SEARCH_NODES} recursion "
                "nodes without deciding"
            ),
        )

    if decided:
        emb = SubgraphEmbedding(
            mapping=tuple((pv, mapping[pv]) for pv in pattern_vertices),
        )
        return SubgraphPatternResult(
            host_graph=request.host,
            pattern_graph=request.pattern,
            host_vertex_count=request.host.vertex_count,
            pattern_vertex_count=request.pattern.vertex_count,
            exists=True,
            embedding=emb,
        )

    return SubgraphPatternResult(
        host_graph=request.host,
        pattern_graph=request.pattern,
        host_vertex_count=request.host.vertex_count,
        pattern_vertex_count=request.pattern.vertex_count,
        exists=False,
    )


class _BudgetExceededError(Exception):
    """Internal control flow: recursion node budget exhausted."""


__all__ = [
    "decide_fixed_length_cycle",
    "find_subgraph_pattern",
]
