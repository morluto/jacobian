"""Cycle-length profile kernel."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import rfc8785

from jacobian.canonical import CanonicalizationError, CanonicalLimits
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
    """Count simple-path prefixes with an early work cutoff."""
    vertex_count = len(graph.vertices)
    vertex_to_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    adjacency = [[False] * vertex_count for _ in range(vertex_count)]
    for left, right in graph.edges:
        left_index = vertex_to_index[left]
        right_index = vertex_to_index[right]
        adjacency[left_index][right_index] = True
        adjacency[right_index][left_index] = True

    work = 0
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

    try:
        result_bytes = simple_undirected_graph_wire_bytes(graph) + 256
    except CanonicalizationError:
        _reject(
            "cycle_profile.result_bound",
            "the complete cycle profile exceeds the canonical output bound",
        )
    if graph.edges:
        # The transport path NFC-normalizes strings and RFC-8785 escapes control
        # characters, so raw UTF-8 lengths undercount the actual result.
        label_bytes = sum(
            len(rfc8785.dumps(unicodedata.normalize("NFC", label)))
            for label in graph.vertices
        )
        # A simple cycle of length k consumes k distinct edges.  Charging only
        # lengths that can occur avoids rejecting sparse graphs whose exact
        # profile is empty (for example, a one-edge graph).
        max_cycle_length = min(vertex_count, len(graph.edges))
        for length in range(3, max_cycle_length + 1):
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
