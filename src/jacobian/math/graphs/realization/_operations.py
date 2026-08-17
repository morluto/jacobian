"""Domain-owned graph realization operations."""

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
    """Erdos-Gallai theorem: a degree sequence is graphical iff
    sum(degrees) is even and for all k in 1..n:
        sum(d_i for i <= k) <= k*(k-1) + sum(min(d_i, k) for i > k).
    """
    if any(d < 0 for d in degrees):
        return False
    if sum(degrees) % 2 != 0:
        return False
    n = len(degrees)
    sorted_deg = sorted(degrees, reverse=True)
    if any(d >= n for d in sorted_deg):
        return False
    cumulative = 0
    for k in range(1, n + 1):
        cumulative += sorted_deg[k - 1]
        rhs = k * (k - 1)
        for i in range(k, n):
            rhs += min(sorted_deg[i], k)
        if cumulative > rhs:
            return False
    return True


def compute_degree_sequence(
    request: DegreeSequenceRequest,
) -> DegreeSequenceResult:
    """Determine if a degree sequence is graphical using the Erdos-Gallai theorem."""
    degrees = request.sequence.degrees
    is_graphical = _is_graphical_erdos_gallai(degrees)
    return DegreeSequenceResult(
        is_graphical=is_graphical,
        degree_sum=sum(degrees),
        vertex_count=len(degrees),
    )


def compute_graph_realization(
    request: GraphRealizationRequest,
) -> GraphRealizationResult:
    """Construct a simple graph realizing the degree sequence using Havel-Hakimi."""
    degrees = request.sequence.degrees

    if not _is_graphical_erdos_gallai(degrees):
        return GraphRealizationResult(
            is_graphical=False,
            vertex_count=len(degrees),
            edges=(),
        )

    g = nx.havel_hakimi_graph(degrees)  # type: ignore[call-overload]
    edges = tuple(tuple(edge) for edge in g.edges())
    return GraphRealizationResult(
        is_graphical=True,
        vertex_count=len(degrees),
        edges=edges,
    )


def compute_graphicality_check(
    request: GraphicalityCheckRequest,
) -> GraphicalityCheckResult:
    """Check if a degree sequence is graphical, providing a certificate.

    The certificate is the Erdos-Gallai inequality state when the sequence is
    non-graphical, or the string 'ERDOS-GALLAI' when it is graphical.
    """
    degrees = request.sequence.degrees
    n = len(degrees)
    degree_sum = sum(degrees)

    if degree_sum % 2 != 0:
        certificate = "odd-sum: the degree sum is not even"
        return GraphicalityCheckResult(
            is_graphical=False,
            degree_sum=degree_sum,
            vertex_count=n,
            certificate=certificate,
        )

    sorted_deg = sorted(degrees, reverse=True)
    if any(d >= n for d in sorted_deg):
        bad = next(d for d in sorted_deg if d >= n)
        certificate = f"degree {bad} exceeds vertex count {n - 1}"
        return GraphicalityCheckResult(
            is_graphical=False,
            degree_sum=degree_sum,
            vertex_count=n,
            certificate=certificate,
        )

    cumulative = 0
    for k in range(1, n + 1):
        cumulative += sorted_deg[k - 1]
        rhs = k * (k - 1)
        for i in range(k, n):
            rhs += min(sorted_deg[i], k)
        if cumulative > rhs:
            certificate = (
                f"erdos-gallai violation at k={k}: left={cumulative} > right={rhs}"
            )
            return GraphicalityCheckResult(
                is_graphical=False,
                degree_sum=degree_sum,
                vertex_count=n,
                certificate=certificate,
            )

    return GraphicalityCheckResult(
        is_graphical=True,
        degree_sum=degree_sum,
        vertex_count=n,
        certificate="ERDOS-GALLAI",
    )


def compute_realization_check(
    request: RealizationCheckRequest,
) -> RealizationCheckResult:
    """Verify that a graph realizes a given degree sequence."""
    degrees = request.sequence.degrees
    edges = request.graph.edges
    vertex_count = request.graph.vertex_count

    g: nx.Graph[Any] = nx.Graph()
    g.add_nodes_from(range(vertex_count))
    for source, target in edges:
        g.add_edge(source, target)

    actual = tuple(len(g[node]) for node in range(vertex_count))
    return RealizationCheckResult(
        is_realization=actual == degrees,
        expected_degrees=degrees,
        actual_degrees=actual,
    )
