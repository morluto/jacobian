"""Equitable k-colouring kernel."""

from __future__ import annotations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring.equitable_k_coloring._models import (
    MAX_EQUITABLE_COLORING_SEARCH_NODES,
    EquitableColoringAssignment,
    EquitableColoringResult,
    _is_complete,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["decide_equitable_k_coloring", "verify_equitable_coloring"]


def _admit(graph: SimpleUndirectedGraph, k: int) -> None:
    """Admit the bounded search once at the native operation boundary."""
    if k <= 0:
        raise OperationDomainValidationError(
            location=("k",),
            code="graph.equitable_coloring_positive_palette",
            message="equitable coloring requires a positive palette size",
        )
    n = len(graph.vertices)
    if (
        graph.edges
        and 0 < k < n
        and not _is_complete(graph)
        and k**n > MAX_EQUITABLE_COLORING_SEARCH_NODES
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


def _direct_result(
    graph: SimpleUndirectedGraph,
    k: int,
) -> EquitableColoringResult | None:
    n = len(graph.vertices)
    if _is_complete(graph) and k < n:
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


equitable_coloring_check = verify_equitable_coloring
