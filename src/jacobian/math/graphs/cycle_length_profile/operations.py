"""Cycle-length profile kernel."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import networkx as nx
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
    search_vertices: tuple[str, ...] | None = None


def _maximum_path_work(graph: SimpleUndirectedGraph) -> int:
    """Count simple-path prefixes with an early work cutoff."""
    vertex_count = len(graph.vertices)
    if len(graph.edges) == vertex_count * (vertex_count - 1) // 2:
        # In a complete graph the first DFS branch witnesses every length.
        return vertex_count**3
    if _is_wheel_graph(graph):
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


def _is_wheel_graph(graph: SimpleUndirectedGraph) -> bool:
    """Recognize a wheel topology before applying the first-witness bound."""
    vertex_count = len(graph.vertices)
    if vertex_count < 4 or len(graph.edges) != 2 * (vertex_count - 1):
        return False
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
        return False
    rim = [vertex for vertex in graph.vertices if vertex != hubs[0]]
    if not all(len(adjacency[vertex]) == 3 for vertex in rim):
        return False
    rim_adjacency = {vertex: adjacency[vertex] - {hubs[0]} for vertex in rim}
    if not all(len(neighbors) == 2 for neighbors in rim_adjacency.values()):
        return False
    seen = {rim[0]}
    stack = [rim[0]]
    while stack:
        vertex = stack.pop()
        for neighbor in rim_adjacency[vertex] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return len(seen) == len(rim)


def _wheel_search_order(graph: SimpleUndirectedGraph) -> tuple[str, ...] | None:
    """Return a hub-then-cyclic-rim order for the wheel DFS shortcut."""
    if not _is_wheel_graph(graph):
        return None
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
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


def _cycle_block_feasible_lengths(
    graph: SimpleUndirectedGraph,
) -> list[tuple[frozenset[int], list[int]]]:
    """Return conservative cycle lengths and their supporting block sizes.

    A block that is itself a simple cycle contributes only its full length;
    charging every shorter row for it needlessly reserves output space for
    witnesses the kernel cannot produce.
    """
    topology: nx.Graph[str] = nx.Graph()
    topology.add_nodes_from(graph.vertices)
    topology.add_edges_from(graph.edges)
    feasible: list[tuple[frozenset[int], list[int]]] = []
    for raw_block in nx.biconnected_components(topology):
        block = cast(set[str], raw_block)
        if len(block) < 3:
            continue
        edge_count = sum(
            left in block and right in block for left, right in graph.edges
        )
        degrees = {
            vertex: sum(
                edge[0] in block and edge[1] in block and vertex in edge
                for edge in graph.edges
                if edge[0] != edge[1]
            )
            for vertex in block
        }
        if edge_count == len(block) and all(degree == 2 for degree in degrees.values()):
            lengths: Iterable[int] = (len(block),)
        elif nx.is_bipartite(topology.subgraph(block)):
            coloring = nx.bipartite.color(topology.subgraph(block))
            part_sizes = (
                sum(not color for color in coloring.values()),
                sum(color for color in coloring.values()),
            )
            maximum_length = min(edge_count, 2 * min(part_sizes))
            lengths = range(4, maximum_length + 1, 2)
        else:
            lengths = range(3, min(len(block), edge_count) + 1)
        label_sizes = sorted(
            (
                len(rfc8785.dumps(unicodedata.normalize("NFC", label)))
                for label in block
            ),
            reverse=True,
        )
        feasible.append((frozenset(lengths), label_sizes))
    return feasible


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
        block_feasible_lengths = _cycle_block_feasible_lengths(graph)
        witness_label_bytes_by_length: dict[int, int] = {}
        for lengths, label_sizes in block_feasible_lengths:
            for length in lengths:
                witness_label_bytes_by_length[length] = max(
                    witness_label_bytes_by_length.get(length, 0),
                    sum(label_sizes[:length]),
                )
        for length in sorted(witness_label_bytes_by_length):
            witness_label_bytes = witness_label_bytes_by_length[length]
            result_bytes += 32 + witness_label_bytes + 2 * length
    if result_bytes > MAX_RESULT_BYTES:
        _reject(
            "cycle_profile.result_bound",
            "the complete cycle profile exceeds the canonical output bound",
        )
    return _AdmissionPlan(graph=graph, search_vertices=_wheel_search_order(graph))


def compute_cycle_length_profile(
    graph: SimpleUndirectedGraph,
) -> CycleLengthProfileResult:
    """Return the complete cycle-length profile of a simple graph.

    For each length k from 3 to |V|, check if the graph contains a simple
    k-cycle. Return one canonical witness cycle for each present length.
    """
    plan = _admit(graph)
    vertices = list(plan.search_vertices or plan.graph.vertices)

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
