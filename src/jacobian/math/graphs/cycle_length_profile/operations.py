"""Cycle-length profile kernel."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.cycle_length_profile._models import (
    MAX_VERTICES,
    CycleLengthProfileResult,
    CycleLengthRow,
)
from jacobian.math.graphs.values import (
    SimpleUndirectedGraph,
    simple_undirected_graph_wire_bytes,
)

__all__ = ["compute_cycle_length_profile"]

MAX_SEARCH_WORK = 10_000_000
MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    graph: SimpleUndirectedGraph


def _maximum_path_work(graph: SimpleUndirectedGraph) -> int:
    """Bound DFS paths from the graph's maximum branching factor."""
    vertex_count = len(graph.vertices)
    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    maximum_degree = max(degrees.values(), default=0)
    work = 0
    for length in range(3, vertex_count + 1):
        for start in range(vertex_count):
            available = min(maximum_degree, vertex_count - start - 1)
            if available < length - 1:
                continue
            paths = 1
            for offset in range(length - 1):
                paths *= available - offset
            work += paths * length * length
    return work


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
    if _maximum_path_work(graph) > MAX_SEARCH_WORK:
        _reject(
            "cycle_profile.work_bound",
            "complete cycle-profile search exceeds the admitted work bound",
        )

    label_bytes = sum(len(label.encode("utf-8")) + 2 for label in graph.vertices)
    result_bytes = simple_undirected_graph_wire_bytes(graph) + 256
    for length in range(3, vertex_count + 1):
        result_bytes += 32 + length * (label_bytes + 2)
    if result_bytes > MAX_RESULT_BYTES:
        _reject(
            "cycle_profile.result_bound",
            "the complete cycle profile exceeds the canonical output bound",
        )
    return _AdmissionPlan(graph=graph)


def compute_cycle_length_profile(
    graph: SimpleUndirectedGraph,
) -> CycleLengthProfileResult:
    """Return the complete cycle-length profile of a simple graph.

    For each length k from 3 to |V|, check if the graph contains a simple
    k-cycle. Return one canonical witness cycle for each present length.
    """
    plan = _admit(graph)
    vertices = list(plan.graph.vertices)

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

    rows = [CycleLengthRow(cycle_length=k, witness=w) for k, w in sorted(found.items())]
    return CycleLengthProfileResult(graph=graph, rows=tuple(rows))


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
        visited: list[int],
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
            if nxt not in visited and adj_matrix[current][nxt]:
                visited.append(nxt)
                result = dfs(start, nxt, visited, [*path, nxt])
                if result is not None:
                    return result
                visited.pop()
        return None

    for start in range(n):
        result = dfs(start, start, [start], [start])
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
