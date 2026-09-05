"""Kernel for edge-deletion diameter profile."""

from __future__ import annotations

import time

import networkx as nx

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs._networkx import graph_from_value
from jacobian.math.graphs.edge_deletion_diameter_profile._models import (
    EdgeDeletionDiameterEntry,
    EdgeDeletionDiameterProfileResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_DELETION_DIAMETER_WORK = 50_000_000
MAX_RETAINED_LABEL_CHARACTERS = 1_000_000
_OWNER_DEADLINE_SECONDS = 3600.0


def _diameter_profile_work(vertex_count: int, edge_count: int) -> int:
    """Charge all-sources BFS for the source graph and every single-edge deletion.

    NetworkX diameter computes eccentricities by shortest paths from every
    vertex, so one call is ``n·(n+m)`` and the profile invokes it ``m+1``
    times.
    """

    return (edge_count + 1) * vertex_count * (vertex_count + edge_count)


def _retained_label_characters(graph: SimpleUndirectedGraph) -> int:
    source_label_characters = sum(map(len, graph.vertices)) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    entry_label_characters = sum(len(left) + len(right) for left, right in graph.edges)
    return source_label_characters + entry_label_characters


def _admit(graph: SimpleUndirectedGraph) -> nx.Graph[str]:
    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    if vertex_count == 0:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion_diameter.empty_graph",
            message="graph must be nonempty",
        )
    if _retained_label_characters(graph) > MAX_RETAINED_LABEL_CHARACTERS:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.edge_deletion_diameter.retained_labels_exceed_bound",
            message=(
                "edge-deletion diameter profile source and entry labels exceed "
                "the retained-character bound"
            ),
        )
    work = _diameter_profile_work(vertex_count, edge_count)
    if work > MAX_EDGE_DELETION_DIAMETER_WORK:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion_diameter.work_exceeds_bound",
            message=(
                "edge-deletion diameter profile exceeds the "
                f"{MAX_EDGE_DELETION_DIAMETER_WORK}-unit all-sources BFS work bound"
            ),
        )
    backend_graph = graph_from_value(graph)
    if not nx.is_connected(backend_graph):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion_diameter.not_connected",
            message="graph must be connected",
        )
    return backend_graph


def edge_deletion_diameter_profile(
    graph: SimpleUndirectedGraph,
) -> EdgeDeletionDiameterProfileResult:
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return edge_deletion_diameter_profile(graph)
    if execution.deadline is None:
        bind_request_deadline(execution.started_at + _OWNER_DEADLINE_SECONDS)
    request_checkpoint("before edge-deletion diameter admission")
    backend_graph = _admit(graph)
    source_diameter = int(nx.diameter(backend_graph))
    entries: list[EdgeDeletionDiameterEntry] = []
    for idx in range(len(graph.edges)):
        request_checkpoint("during edge-deletion diameter profile")
        u, v = graph.edges[idx]
        remaining = backend_graph.copy()
        if remaining.has_edge(u, v):
            remaining.remove_edge(u, v)
        elif remaining.has_edge(v, u):
            remaining.remove_edge(v, u)
        else:
            raise OperationDomainValidationError(
                location=("graph", "edges", idx),
                code="graph.edge_deletion_diameter.edge_not_found",
                message=f"edge {(u, v)} not found in NetworkX graph",
            )
        edge: tuple[str, str] = graph.edges[idx]
        if not nx.is_connected(remaining):
            entries.append(
                EdgeDeletionDiameterEntry(
                    edge=edge,
                    edge_index=idx,
                    result="DISCONNECTED",
                    diameter=None,
                )
            )
        else:
            diam = int(nx.diameter(remaining))
            entries.append(
                EdgeDeletionDiameterEntry(
                    edge=edge,
                    edge_index=idx,
                    result="DIAMETER",
                    diameter=diam,
                )
            )
    return EdgeDeletionDiameterProfileResult._from_kernel(
        graph=graph,
        source_diameter=source_diameter,
        entries=tuple(entries),
    )


__all__ = [
    "MAX_EDGE_DELETION_DIAMETER_WORK",
    "MAX_RETAINED_LABEL_CHARACTERS",
    "edge_deletion_diameter_profile",
]
