"""Cycle-length profile kernel."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.cycle_length_profile._models import (
    MAX_VERTICES,
    CycleLengthProfileResult,
    CycleLengthRow,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_cycle_length_profile"]

MAX_SEARCH_WORK = 10_000_000


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    graph: SimpleUndirectedGraph
    wheel_order: tuple[str, ...] | None = None


def _maximum_path_work(graph: SimpleUndirectedGraph, *, is_wheel: bool = False) -> int:
    """Count simple-path prefixes with an early work cutoff."""
    vertex_count = len(graph.vertices)
    if len(graph.edges) == vertex_count * (vertex_count - 1) // 2:
        # In a complete graph the first DFS branch witnesses every length.
        return vertex_count**3
    if is_wheel:
        # A wheel is pancyclic: the hub plus a contiguous rim segment gives
        # every length from three through n.  The kernel's first-witness DFS
        # reaches each such segment after at most O(n^2) neighbor checks, so
        # charge one cubic envelope instead of all simple paths.
        return vertex_count**3
    vertex_to_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    adjacency: list[list[bool]] = [[False] * vertex_count for _ in range(vertex_count)]
    for left, right in graph.edges:
        left_index = vertex_to_index[left]
        right_index = vertex_to_index[right]
        adjacency[left_index][right_index] = True
        adjacency[right_index][left_index] = True

    # The kernel repeats root and one-edge scans for every target length, even
    # when no path reaches depth three (for example, a perfect matching).
    root_scan_candidates = vertex_count * (vertex_count - 1) // 2 + 2 * len(graph.edges)
    work = root_scan_candidates * max(1, vertex_count - 2)
    # Each root edge accepted by ``nxt >= start + 1`` causes the kernel to
    # scan the complete remaining suffix once for every target length, even
    # when that edge has no continuation. Charge this depth-two work instead
    # of treating every edge as a constant-cost branch.
    depth_two_candidates = 0
    for left, right in graph.edges:
        left_index = vertex_to_index[left]
        right_index = vertex_to_index[right]
        depth_two_candidates += vertex_count - min(left_index, right_index) - 1
    work += depth_two_candidates * max(1, vertex_count - 2)
    for start in range(vertex_count):
        visited = {start}

        def visit(current: int, visited: set[int] = visited) -> None:
            nonlocal work
            for nxt in range(vertex_count):
                if nxt in visited or not adjacency[current][nxt]:
                    continue
                visited.add(nxt)
                if len(visited) >= 3:
                    work += vertex_count * vertex_count
                    if work > MAX_SEARCH_WORK:
                        return
                visit(nxt)
                visited.remove(nxt)
                if work > MAX_SEARCH_WORK:
                    return

        visit(start)
        if work > MAX_SEARCH_WORK:
            return work
    return work


def _wheel_search_order(graph: SimpleUndirectedGraph) -> tuple[str, ...] | None:
    """Return a hub-then-cyclic-rim order for the wheel DFS shortcut.

    Recognizes the wheel topology and returns the cyclic rim order in one pass,
    so callers that need both recognition and the order do not re-derive it.
    """
    vertex_count = len(graph.vertices)
    if vertex_count < 4 or len(graph.edges) != 2 * (vertex_count - 1):
        return None
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    hubs = [
        vertex
        for vertex, neighbors in adjacency.items()
        if len(neighbors) == vertex_count - 1
    ]
    if len(hubs) != 1:
        return None
    rim = [vertex for vertex in graph.vertices if vertex != hubs[0]]
    if not all(len(adjacency[vertex]) == 3 for vertex in rim):
        return None
    rim_adjacency = {vertex: adjacency[vertex] - {hubs[0]} for vertex in rim}
    if not all(len(neighbors) == 2 for neighbors in rim_adjacency.values()):
        return None
    seen = {rim[0]}
    stack = [rim[0]]
    while stack:
        vertex = stack.pop()
        for neighbor in rim_adjacency[vertex] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    if len(seen) != len(rim):
        return None
    hub = next(
        vertex
        for vertex, neighbors in adjacency.items()
        if len(neighbors) == len(graph.vertices) - 1
    )
    rim = sorted(vertex for vertex in graph.vertices if vertex != hub)
    rim_adjacency = {vertex: adjacency[vertex] - {hub} for vertex in rim}
    order = [rim[0]]
    previous: str | None = None
    current = rim[0]
    while len(order) < len(rim):
        candidates = sorted(
            neighbor for neighbor in rim_adjacency[current] if neighbor != previous
        )
        next_vertex = candidates[0] if candidates[0] not in order else candidates[1]
        order.append(next_vertex)
        previous, current = current, next_vertex
    return (hub, *order)


def _cycle_core_vertices(graph: SimpleUndirectedGraph) -> set[str]:
    """Return vertices in the graph's cycle-bearing 2-core."""
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    pending = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) < 2]
    removed = set(pending)
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor in removed:
                continue
            adjacency[neighbor].discard(vertex)
            if len(adjacency[neighbor]) < 2:
                removed.add(neighbor)
                pending.append(neighbor)
    remaining = set(adjacency) - removed
    core_vertices: set[str] = set()
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        core_vertices.update(component)
    return core_vertices


def _reject(code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("graph",), code=code, message=message
    )


def _admit(graph: SimpleUndirectedGraph) -> _AdmissionPlan:
    """Validate native and MCP graph, work, and result envelopes once."""
    if not isinstance(graph, SimpleUndirectedGraph):
        _reject(
            "cycle_profile.graph_type",
            "graph must be a canonical simple undirected graph",
        )
    vertex_count = len(graph.vertices)
    if vertex_count > MAX_VERTICES:
        _reject(
            "cycle_profile.vertex_bound",
            f"cycle profiles support at most {MAX_VERTICES} vertices",
        )
    wheel_order = _wheel_search_order(graph)
    if _maximum_path_work(graph, is_wheel=wheel_order is not None) > MAX_SEARCH_WORK:
        _reject(
            "cycle_profile.work_bound",
            "complete cycle-profile search exceeds the admitted work bound",
        )

    return _AdmissionPlan(graph=graph, wheel_order=wheel_order)


def compute_cycle_length_profile(
    graph: SimpleUndirectedGraph,
) -> CycleLengthProfileResult:
    """Return the complete cycle-length profile of a simple graph.

    For each length k from 3 to |V|, check if the graph contains a simple
    k-cycle. Return one canonical witness cycle for each present length.
    """
    plan = _admit(graph)
    vertices = list(plan.wheel_order or plan.graph.vertices)

    n = len(vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    adj_matrix = [[False] * n for _ in range(n)]
    for a, b in graph.edges:
        i, j = vertex_to_idx[a], vertex_to_idx[b]
        adj_matrix[i][j] = True
        adj_matrix[j][i] = True

    found: dict[int, tuple[str, ...]] = {}
    for length in range(3, n + 1):
        witness = _find_cycle_of_length(length, n, adj_matrix, vertices)
        if witness is not None:
            found[length] = witness

    rows = [
        CycleLengthRow._from_kernel(cycle_length=k, witness=w)
        for k, w in sorted(found.items())
    ]
    return CycleLengthProfileResult._from_kernel(graph, tuple(rows))


def _find_cycle_of_length(
    length: int,
    n: int,
    adj_matrix: list[list[bool]],
    vertices: list[str],
) -> tuple[str, ...] | None:
    """Find one simple cycle of the given length using DFS backtracking."""

    def dfs(
        start: int,
        current: int,
        visited: list[bool],
        path: list[int],
    ) -> tuple[str, ...] | None:
        if len(path) == length:
            if adj_matrix[current][start]:
                return tuple(vertices[i] for i in path)
            return None
        # The initial vertex is the minimum index in the cycle.  All other
        # vertices are therefore eligible at every depth; restricting them to
        # ``current + 1`` misses cycles whose indices go down and then up.
        for nxt in range(start + 1, n):
            if not visited[nxt] and adj_matrix[current][nxt]:
                visited[nxt] = True
                path.append(nxt)
                result = dfs(start, nxt, visited, path)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None

    for start in range(n):
        visited = [False] * n
        visited[start] = True
        result = dfs(start, start, visited, [start])
        if result is not None:
            canonical = _canonicalize_cycle(result)
            return canonical
    return None


def _canonicalize_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Return the lexicographically smallest rotation in either orientation."""
    n = len(cycle)
    rotations = [cycle[i:] + cycle[:i] for i in range(n)]
    reversed_cycle = (cycle[0], *reversed(cycle[1:]))
    rotations.extend(reversed_cycle[i:] + reversed_cycle[:i] for i in range(n))
    return min(rotations)
