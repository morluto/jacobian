"""Exact chordal recognition with elimination and cycle certificates."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.chordal._models import (
    MAX_CHORDAL_CERTIFICATE_WORK,
    MAX_CHORDAL_ORDER_WORK,
    ChordalRecognitionResult,
    ChordalStatus,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["recognize_chordal", "verify_chordality"]


def verify_chordality(claim: ChordalRecognitionResult) -> bool:
    """Check the PEO or chordless-cycle relation without recognition replay.

    PEO work shares recognition's degree-square admission. Checking a supplied
    cycle costs at most 256 squared adjacency lookups; no cycle search runs.
    """
    neighbors, edge_count = _adjacency(claim.graph)
    indices = {label: i for i, label in enumerate(claim.graph.vertices)}
    if claim.status == "CHORDAL":
        _admit_recognition(claim.graph, [len(row) for row in neighbors], edge_count)
        ordering = tuple(indices[label] for label in claim.elimination_ordering)
        return _later_clique_failure(ordering, neighbors) is None
    cycle = tuple(indices[label] for label in claim.induced_cycle)
    for i, first in enumerate(cycle):
        for j in range(i + 1, len(cycle)):
            consecutive = j == i + 1 or (i == 0 and j == len(cycle) - 1)
            if (cycle[j] in neighbors[first]) != consecutive:
                return False
    return True


def _reject(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("graph",), code=code, message=message
    )


def _adjacency(
    graph: SimpleUndirectedGraph,
) -> tuple[list[set[int]], int]:
    """Return index adjacency and the edge count."""

    position = {vertex: index for index, vertex in enumerate(graph.vertices)}
    neighbors: list[set[int]] = [set() for _ in graph.vertices]
    for left, right in graph.edges:
        neighbors[position[left]].add(position[right])
        neighbors[position[right]].add(position[left])
    return neighbors, len(graph.edges)


def _maximum_cardinality_ordering(
    order: int, neighbors: list[set[int]]
) -> tuple[int, ...]:
    """Number vertices by maximum already-numbered neighbors (MCS).

    Ties resolve toward the smallest declared position, so the ordering is
    deterministic. A maximum-cardinality-search ordering of a chordal graph
    is a perfect elimination ordering (Tarjan-Yannakakis); hence an
    MCS-ordering failure certifies non-chordality.
    """

    weight = [0] * order
    numbered = [False] * order
    result: list[int] = []
    for _ in range(order):
        pick = -1
        for index in range(order):
            if not numbered[index] and (
                pick < 0
                or weight[index] > weight[pick]
                or (weight[index] == weight[pick] and index < pick)
            ):
                pick = index
        numbered[pick] = True
        result.append(pick)
        for neighbor in neighbors[pick]:
            if not numbered[neighbor]:
                weight[neighbor] += 1
    # First-picked vertices number highest, so the picking order reverses
    # into the candidate perfect elimination ordering.
    return tuple(reversed(result))


def _later_clique_failure(
    ordering: tuple[int, ...], neighbors: list[set[int]]
) -> tuple[int, int, int] | None:
    """Return (v, x, y) with x, y nonadjacent later-neighbors of v, if any."""

    position_of = {vertex: rank for rank, vertex in enumerate(ordering)}
    for rank, vertex in enumerate(ordering):
        later = sorted(
            neighbor for neighbor in neighbors[vertex] if position_of[neighbor] > rank
        )
        for left_index in range(len(later)):
            for right_index in range(left_index + 1, len(later)):
                if later[right_index] not in neighbors[later[left_index]]:
                    return vertex, later[left_index], later[right_index]
    return None


def _shortest_avoiding_path(
    start: int,
    goal: int,
    forbidden: set[int],
    neighbors: list[set[int]],
) -> tuple[int, ...] | None:
    """Shortest start-goal path avoiding forbidden vertices (BFS)."""

    if start == goal:
        return (start,)
    previous: dict[int, int | None] = {start: None}
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(neighbors[current]):
            if neighbor in forbidden or neighbor in previous:
                continue
            previous[neighbor] = current
            if neighbor == goal:
                path = [goal]
                while previous[path[-1]] is not None:
                    previous_vertex = previous[path[-1]]
                    assert previous_vertex is not None
                    path.append(previous_vertex)
                return tuple(reversed(path))
            queue.append(neighbor)
    return None


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Rotate to the smallest label with the smaller second endpoint."""

    start = cycle.index(min(cycle))
    rotated = cycle[start:] + cycle[:start]
    flipped = (rotated[0], *tuple(reversed(rotated[1:])))
    return min(rotated, flipped)


def _induced_cycle(
    order: int,
    neighbors: list[set[int]],
    labels: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract an ordered induced cycle of length at least four.

    For a center v with nonadjacent neighbors x, y, a shortest x-y path
    avoiding v and every other neighbor of v closes an induced cycle
    through v: the path is chordless by minimality and its interior avoids
    v's neighborhood by construction. A non-chordal graph always holds such
    a triple on one of its induced cycles, so the exhaustive scan succeeds.
    """

    for center in range(order):
        center_neighbors = sorted(neighbors[center])
        for left_index in range(len(center_neighbors)):
            for right_index in range(left_index + 1, len(center_neighbors)):
                first, second = (
                    center_neighbors[left_index],
                    center_neighbors[right_index],
                )
                if second in neighbors[first]:
                    continue
                forbidden = (set(center_neighbors) | {center}) - {first, second}
                path = _shortest_avoiding_path(first, second, forbidden, neighbors)
                if path is None:
                    continue
                cycle = (center, *path)
                return _canonical_cycle(tuple(labels[vertex] for vertex in cycle))
    raise AssertionError("induced-cycle extraction exhausted all triples")


def _admit_recognition(
    graph: SimpleUndirectedGraph,
    neighbor_counts: list[int],
    edge_count: int,
) -> None:
    order = len(graph.vertices)
    order_work = order * edge_count + sum(count * count for count in neighbor_counts)
    if order_work > MAX_CHORDAL_ORDER_WORK:
        raise _reject(
            "graph.chordal.order_work_bound",
            "chordal ordering verification exceeds the "
            f"{MAX_CHORDAL_ORDER_WORK:,}-unit work bound",
        )


def _admit_certificate(neighbor_counts: list[int], order: int, edge_count: int) -> None:
    pair_work = sum(count * (count - 1) // 2 for count in neighbor_counts)
    if pair_work * (order + edge_count) > MAX_CHORDAL_CERTIFICATE_WORK:
        raise _reject(
            "graph.chordal.certificate_work_bound",
            "induced-cycle extraction exceeds the "
            f"{MAX_CHORDAL_CERTIFICATE_WORK:,}-unit work bound",
        )


def recognize_chordal(graph: SimpleUndirectedGraph) -> ChordalRecognitionResult:
    """Recognize chordality with an elimination ordering or induced cycle.

    A maximum-cardinality-search ordering is verified as a perfect
    elimination ordering; success is CHORDAL. Otherwise the graph is
    certifiably non-chordal and a shortest-avoiding-path search extracts an
    ordered induced cycle of length at least four. Ordering verification and
    certificate extraction carry independent work budgets.
    """

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("recognize_chordal expects a SimpleUndirectedGraph")
    index_neighbors, edge_count = _adjacency(graph)
    order = len(graph.vertices)
    _admit_recognition(graph, [len(peers) for peers in index_neighbors], edge_count)
    ordering = _maximum_cardinality_ordering(order, index_neighbors)
    if _later_clique_failure(ordering, index_neighbors) is None:
        return ChordalRecognitionResult._from_kernel(
            graph=graph,
            status="CHORDAL",
            elimination_ordering=tuple(graph.vertices[vertex] for vertex in ordering),
        )
    _admit_certificate([len(peers) for peers in index_neighbors], order, edge_count)
    cycle = _induced_cycle(order, index_neighbors, graph.vertices)
    status: ChordalStatus = "NONCHORDAL"
    return ChordalRecognitionResult._from_kernel(
        graph=graph, status=status, induced_cycle=cycle
    )
