"""Equitable k-colouring kernel."""

from __future__ import annotations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring.equitable_k_coloring._models import (
    MAX_EQUITABLE_COLORING_SEARCH_DEPTH,
    MAX_EQUITABLE_COLORING_SEARCH_NODES,
    EquitableColoringAssignment,
    EquitableColoringResult,
    _is_complete,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["decide_equitable_k_coloring", "verify_equitable_coloring"]


def _admit(graph: SimpleUndirectedGraph, k: int) -> None:
    """Admit the bounded generic-search or bipartite-DP envelope once."""
    if k <= 0:
        raise OperationDomainValidationError(
            location=("k",),
            code="graph.equitable_coloring_positive_palette",
            message="equitable coloring requires a positive palette size",
        )
    n = len(graph.vertices)
    needs_search = bool(graph.edges) and 0 < k < n and not _is_complete(graph) and k > 1
    if needs_search and (
        n > MAX_EQUITABLE_COLORING_SEARCH_DEPTH
        or (k != 2 and k**n > MAX_EQUITABLE_COLORING_SEARCH_NODES)
    ):
        raise OperationDomainValidationError(
            location=("graph", "k"),
            code="graph.equitable_coloring_search_exceeded",
            message="equitable coloring exceeds the 1000000-node search bound",
        )


def _result(
    graph: SimpleUndirectedGraph,
    k: int,
    colorable: bool,
    coloring: tuple[int, ...] | None = None,
) -> EquitableColoringResult:
    assignment = (
        EquitableColoringAssignment(graph=graph, k=k, coloring=coloring)
        if coloring is not None
        else None
    )
    return EquitableColoringResult(
        graph=graph, k=k, colorable=colorable, coloring=assignment
    )


def decide_equitable_k_coloring(
    graph: SimpleUndirectedGraph,
    k: int,
) -> EquitableColoringResult:
    """Decide whether G has a proper k-colouring with balanced class sizes.

    Every colour class must have size floor(|V|/k) or ceil(|V|/k).
    """
    _admit(graph, k)
    direct_result = _direct_result(graph, k)
    if direct_result is not None:
        return direct_result
    if k == 2:
        return _bipartite_equitable_result(graph)
    n = len(graph.vertices)
    if n > MAX_EQUITABLE_COLORING_SEARCH_DEPTH:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.equitable_coloring_search_depth",
            message=(
                "equitable coloring search supports at most "
                f"{MAX_EQUITABLE_COLORING_SEARCH_DEPTH} vertices"
            ),
        )
    return _backtracking_result(graph, k)


def _backtracking_result(
    graph: SimpleUndirectedGraph, k: int
) -> EquitableColoringResult:
    """Run the admitted generic equitable-colouring search."""
    n = len(graph.vertices)
    nx_graph: nx.Graph[str] = nx.Graph()
    for v in graph.vertices:
        nx_graph.add_node(v)
    for u, v in graph.edges:
        nx_graph.add_edge(u, v)

    vertices = list(graph.vertices)

    base = n // k
    remainder = n % k
    # Classes 0..remainder-1 have size base+1, classes remainder..k-1 have size base

    colors = [-1] * n
    class_sizes = [0] * k

    def backtrack(idx: int) -> list[int] | None:
        if idx == n:
            return list(colors)

        for c in range(k):
            max_size = base + 1 if c < remainder else base
            if class_sizes[c] >= max_size:
                continue

            vertex = vertices[idx]
            ok = True
            for neighbor in nx_graph.neighbors(vertex):
                neighbor_idx = vertices.index(neighbor)
                if colors[neighbor_idx] == c:
                    ok = False
                    break

            if ok:
                colors[idx] = c
                class_sizes[c] += 1
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                colors[idx] = -1
                class_sizes[c] -= 1

        return None

    result_colors = backtrack(0)

    if result_colors is not None:
        return _result(graph, k, True, tuple(result_colors))
    return _result(graph, k, False)


def _bipartite_equitable_result(
    graph: SimpleUndirectedGraph,
) -> EquitableColoringResult:
    """Decide equitable 2-colourability from bipartite component orientations.

    A connected bipartite component has exactly two proper 2-colourings, which
    exchange its two bipartition classes.  Selecting one orientation per
    component is therefore sufficient and necessary.  The dynamic program
    records achievable size-zero classes, using at most ``n * (n + 1)``
    states; the graph carrier bounds ``n`` by the admitted search depth.
    A non-bipartite graph has no proper 2-colouring and therefore returns the
    exact negative decision without entering the generic search.
    """
    vertices = graph.vertices
    index_of = {vertex: index for index, vertex in enumerate(vertices)}
    adjacency: list[list[int]] = [[] for _ in vertices]
    for left, right in graph.edges:
        left_index = index_of[left]
        right_index = index_of[right]
        adjacency[left_index].append(right_index)
        adjacency[right_index].append(left_index)

    parity = [-1] * len(vertices)
    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for start in range(len(vertices)):
        if parity[start] != -1:
            continue
        parity[start] = 0
        stack = [start]
        sides: tuple[list[int], list[int]] = ([], [])
        while stack:
            vertex = stack.pop()
            sides[parity[vertex]].append(vertex)
            for neighbor in adjacency[vertex]:
                if parity[neighbor] == -1:
                    parity[neighbor] = 1 - parity[vertex]
                    stack.append(neighbor)
                elif parity[neighbor] == parity[vertex]:
                    return _result(graph, 2, False)
        components.append((tuple(sides[0]), tuple(sides[1])))

    target_sizes = (len(vertices) // 2, (len(vertices) + 1) // 2)
    reachable_sizes = {0}
    predecessors: list[dict[int, tuple[int, bool]]] = []
    for first_side, second_side in components:
        next_predecessors: dict[int, tuple[int, bool]] = {}
        for size in reachable_sizes:
            next_predecessors.setdefault(size + len(first_side), (size, True))
            next_predecessors.setdefault(size + len(second_side), (size, False))
        predecessors.append(next_predecessors)
        reachable_sizes = set(next_predecessors)

    target = next(
        (target for target in target_sizes if target in reachable_sizes),
        None,
    )
    if target is None:
        return _result(graph, 2, False)

    orientations: list[bool] = []
    for predecessor_map in reversed(predecessors):
        target, first_is_zero = predecessor_map[target]
        orientations.append(first_is_zero)
    orientations.reverse()

    coloring = [0] * len(vertices)
    for (first_side, second_side), first_is_zero in zip(
        components, orientations, strict=True
    ):
        for vertex in first_side:
            coloring[vertex] = 0 if first_is_zero else 1
        for vertex in second_side:
            coloring[vertex] = 1 if first_is_zero else 0
    return _result(graph, 2, True, tuple(coloring))


def _direct_result(
    graph: SimpleUndirectedGraph,
    k: int,
) -> EquitableColoringResult | None:
    n = len(graph.vertices)
    if _is_complete(graph) and k < n:
        return _result(graph, k, False)
    if k == 1 and graph.edges:
        return _result(graph, k, False)
    if k >= n:
        coloring = tuple(range(n))
    elif not graph.edges:
        coloring = tuple(index % k for index in range(n))
    else:
        return None
    return _result(graph, k, True, coloring)


def verify_equitable_coloring(claim: EquitableColoringResult) -> bool:
    """Verify a supplied positive equitable-coloring witness after transport."""
    if not claim.colorable or claim.coloring is None:
        return False
    assignment = claim.coloring
    if assignment.graph != claim.graph or assignment.k != claim.k:
        return False
    if any(
        assignment.coloring[left] == assignment.coloring[right]
        for left, right in (
            (claim.graph.vertices.index(left), claim.graph.vertices.index(right))
            for left, right in claim.graph.edges
        )
    ):
        return False
    class_sizes = [assignment.coloring.count(color) for color in range(claim.k)]
    return max(class_sizes, default=0) - min(class_sizes, default=0) <= 1
