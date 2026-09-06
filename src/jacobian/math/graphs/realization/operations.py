"""Exact native graph realization operations."""

from __future__ import annotations

from typing import Any

import networkx as nx

from jacobian.math.graphs.realization._models import (
    DegreeSequence,
    DegreeSequenceResult,
    GraphicalityCertificate,
    GraphicalityCheckResult,
    GraphRealizationResult,
    RealizationCheckResult,
)
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


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
            min(sorted_degrees[index], k) for index in range(k, vertex_count)
        )
        if cumulative > rhs:
            return False
    return True


def degree_sequence_profile(sequence: DegreeSequence) -> DegreeSequenceResult:
    """Determine if a degree sequence is graphical."""
    degrees = sequence.degrees
    return DegreeSequenceResult(
        sequence=sequence,
        is_graphical=_is_graphical_erdos_gallai(degrees),
    )


def graph_realization(sequence: DegreeSequence) -> GraphRealizationResult:
    """Construct a simple graph realizing the degree sequence."""
    degrees = sequence.degrees
    if not _is_graphical_erdos_gallai(degrees):
        return GraphRealizationResult(sequence=sequence, is_graphical=False)
    graph = nx.havel_hakimi_graph(list(degrees))
    return GraphRealizationResult(
        sequence=sequence,
        is_graphical=True,
        graph=IndexedSimpleUndirectedGraph(
            vertex_count=len(degrees),
            edges=tuple(sorted(tuple(sorted(edge)) for edge in graph.edges())),
        ),
    )


def verify_graph_realization(claim: GraphRealizationResult) -> bool:
    """Return whether a claimed graph realizes its retained degree sequence."""
    is_graphical = _is_graphical_erdos_gallai(claim.sequence.degrees)
    if claim.is_graphical != is_graphical:
        return False
    if not is_graphical:
        return claim.graph is None
    if claim.graph is None or claim.graph.vertex_count != len(claim.sequence.degrees):
        return False
    actual = [0] * claim.graph.vertex_count
    for left, right in claim.graph.edges:
        actual[left] += 1
        actual[right] += 1
    return tuple(actual) == claim.sequence.degrees


def graphicality_check(sequence: DegreeSequence) -> GraphicalityCheckResult:
    """Check graphicality and return a deterministic Erdos-Gallai certificate."""
    degrees = sequence.degrees
    vertex_count = len(degrees)
    degree_sum = sum(degrees)
    if degree_sum % 2:
        return GraphicalityCheckResult(
            sequence=sequence,
            is_graphical=False,
            certificate=GraphicalityCertificate(kind="ODD_SUM"),
        )
    sorted_degrees = sorted(degrees, reverse=True)
    if any(degree >= vertex_count for degree in sorted_degrees):
        bad = next(degree for degree in sorted_degrees if degree >= vertex_count)
        return GraphicalityCheckResult(
            sequence=sequence,
            is_graphical=False,
            certificate=GraphicalityCertificate(
                kind="DEGREE_BOUND", degree=bad, vertex_count=vertex_count
            ),
        )
    cumulative = 0
    for k in range(1, vertex_count + 1):
        cumulative += sorted_degrees[k - 1]
        rhs = k * (k - 1) + sum(
            min(sorted_degrees[index], k) for index in range(k, vertex_count)
        )
        if cumulative > rhs:
            return GraphicalityCheckResult(
                sequence=sequence,
                is_graphical=False,
                certificate=GraphicalityCertificate(
                    kind="ERDOS_GALLAI_VIOLATION",
                    k=k,
                    left=cumulative,
                    right=rhs,
                ),
            )
    return GraphicalityCheckResult(
        sequence=sequence,
        is_graphical=True,
        certificate=GraphicalityCertificate(kind="ERDOS_GALLAI"),
    )


def realization_check(
    graph_value: IndexedSimpleUndirectedGraph,
    sequence: DegreeSequence,
) -> RealizationCheckResult:
    """Verify that a graph realizes a given degree sequence."""
    graph: nx.Graph[Any] = nx.Graph()
    graph.add_nodes_from(range(graph_value.vertex_count))
    graph.add_edges_from(graph_value.edges)
    actual = tuple(len(graph[vertex]) for vertex in range(graph_value.vertex_count))
    return RealizationCheckResult(
        sequence=sequence,
        graph=graph_value,
        is_realization=actual == sequence.degrees,
    )


def verify_degree_sequence_profile(claim: DegreeSequenceResult) -> bool:
    """Return whether a graphicality claim matches its retained sequence."""
    return claim.is_graphical == _is_graphical_erdos_gallai(claim.sequence.degrees)


def verify_graphicality_check(claim: GraphicalityCheckResult) -> bool:
    """Return whether a typed Erdos-Gallai certificate proves the claim."""
    certificate = claim.certificate
    degrees = claim.sequence.degrees
    vertex_count = len(degrees)
    if certificate.kind == "ODD_SUM":
        return not claim.is_graphical and sum(degrees) % 2 == 1
    if certificate.kind == "DEGREE_BOUND":
        return (
            not claim.is_graphical
            and certificate.degree is not None
            and certificate.vertex_count == vertex_count
            and certificate.degree in degrees
            and certificate.degree >= vertex_count
        )
    sorted_degrees = sorted(degrees, reverse=True)
    if certificate.kind == "ERDOS_GALLAI_VIOLATION":
        if (
            claim.is_graphical
            or certificate.k is None
            or certificate.left is None
            or certificate.right is None
            or not 1 <= certificate.k <= vertex_count
        ):
            return False
        k = certificate.k
        left = sum(sorted_degrees[:k])
        right = k * (k - 1) + sum(min(value, k) for value in sorted_degrees[k:])
        return certificate.left == left and certificate.right == right and left > right
    if certificate.kind == "ERDOS_GALLAI":
        if not claim.is_graphical or sum(degrees) % 2:
            return False
        return all(
            sum(sorted_degrees[:k])
            <= k * (k - 1) + sum(min(value, k) for value in sorted_degrees[k:])
            for k in range(1, vertex_count + 1)
        ) and all(value < vertex_count for value in sorted_degrees)
    return False


def verify_realization_check(claim: RealizationCheckResult) -> bool:
    """Return whether a retained graph has the claimed degree sequence."""
    actual = [0] * claim.graph.vertex_count
    for left, right in claim.graph.edges:
        actual[left] += 1
        actual[right] += 1
    return claim.is_realization == (tuple(actual) == claim.sequence.degrees)


__all__ = [
    "degree_sequence_profile",
    "graph_realization",
    "graphicality_check",
    "realization_check",
    "verify_degree_sequence_profile",
    "verify_graph_realization",
    "verify_graphicality_check",
    "verify_realization_check",
]
