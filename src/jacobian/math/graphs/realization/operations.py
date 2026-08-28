"""Exact native graph realization operations."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.math.graphs.realization._models import (
    DegreeSequenceRequest,
    DegreeSequenceResult,
    GraphicalityCheckRequest,
    GraphicalityCheckResult,
    GraphRealizationRequest,
    GraphRealizationResult,
    RealizationCheckRequest,
    RealizationCheckResult,
)


def _is_graphical_erdos_gallai(degrees: tuple[int, ...]) -> bool:
    """Return whether a degree sequence satisfies the Erdos-Gallai theorem."""
    if any(degree < 0 for degree in degrees) or sum(degrees) % 2:
        return False
    vertex_count = len(degrees)
    sorted_degrees = sorted(degrees, reverse=True)
    if any(degree >= vertex_count for degree in sorted_degrees):
        return False
    cumulative = 0
    for k in range(1, vertex_count + 1):
        cumulative += sorted_degrees[k - 1]
        rhs = k * (k - 1) + sum(
            min(sorted_degrees[index], k)
            for index in range(k, vertex_count)
        )
        if cumulative > rhs:
            return False
    return True


def compute_degree_sequence(
    request: DegreeSequenceRequest,
) -> DegreeSequenceResult:
    """Determine if a degree sequence is graphical."""
    degrees = request.sequence.degrees
    return DegreeSequenceResult(
        is_graphical=_is_graphical_erdos_gallai(degrees),
        degree_sum=sum(degrees),
        vertex_count=len(degrees),
    )


def compute_graph_realization(
    request: GraphRealizationRequest,
) -> GraphRealizationResult:
    """Construct a simple graph realizing the degree sequence."""
    degrees = request.sequence.degrees
    if not _is_graphical_erdos_gallai(degrees):
        return GraphRealizationResult(
            is_graphical=False, vertex_count=len(degrees), edges=()
        )
    graph = nx.havel_hakimi_graph(list(degrees))
    return GraphRealizationResult(
        is_graphical=True,
        vertex_count=len(degrees),
        edges=tuple(tuple(edge) for edge in graph.edges()),
    )


def compute_graphicality_check(
    request: GraphicalityCheckRequest,
) -> GraphicalityCheckResult:
    """Check graphicality and return a deterministic Erdos-Gallai certificate."""
    degrees = request.sequence.degrees
    vertex_count = len(degrees)
    degree_sum = sum(degrees)
    if degree_sum % 2:
        certificate = "odd-sum: the degree sum is not even"
        return GraphicalityCheckResult(
            is_graphical=False,
            degree_sum=degree_sum,
            vertex_count=vertex_count,
            certificate=certificate,
        )
    sorted_degrees = sorted(degrees, reverse=True)
    if any(degree >= vertex_count for degree in sorted_degrees):
        bad = next(degree for degree in sorted_degrees if degree >= vertex_count)
        return GraphicalityCheckResult(
            is_graphical=False,
            degree_sum=degree_sum,
            vertex_count=vertex_count,
            certificate=f"degree {bad} exceeds vertex count {vertex_count - 1}",
        )
    cumulative = 0
    for k in range(1, vertex_count + 1):
        cumulative += sorted_degrees[k - 1]
        rhs = k * (k - 1) + sum(
            min(sorted_degrees[index], k)
            for index in range(k, vertex_count)
        )
        if cumulative > rhs:
            return GraphicalityCheckResult(
                is_graphical=False,
                degree_sum=degree_sum,
                vertex_count=vertex_count,
                certificate=(
                    f"erdos-gallai violation at k={k}: "
                    f"left={cumulative} > right={rhs}"
                ),
            )
    return GraphicalityCheckResult(
        is_graphical=True,
        degree_sum=degree_sum,
        vertex_count=vertex_count,
        certificate="ERDOS-GALLAI",
    )


def compute_realization_check(
    request: RealizationCheckRequest,
) -> RealizationCheckResult:
    """Verify that a graph realizes a given degree sequence."""
    graph: nx.Graph[Any] = nx.Graph()
    graph.add_nodes_from(range(request.graph.vertex_count))
    graph.add_edges_from(request.graph.edges)
    actual = tuple(
        len(graph[vertex]) for vertex in range(request.graph.vertex_count)
    )
    return RealizationCheckResult(
        is_realization=actual == request.sequence.degrees,
        expected_degrees=request.sequence.degrees,
        actual_degrees=actual,
    )


__all__ = [
    "compute_degree_sequence",
    "compute_graph_realization",
    "compute_graphicality_check",
    "compute_realization_check",
]
