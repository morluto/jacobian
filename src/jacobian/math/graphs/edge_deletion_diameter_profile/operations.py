"""Kernel for edge-deletion diameter profile."""

from __future__ import annotations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs._networkx import graph_from_value
from jacobian.math.graphs.edge_deletion_diameter_profile._models import (
    MAX_EDGE_DELETION_DIAMETER_EDGES,
    MAX_EDGE_DELETION_DIAMETER_VERTICES,
    EdgeDeletionDiameterEntry,
    EdgeDeletionDiameterProfileResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _admit(graph: SimpleUndirectedGraph) -> None:
    n = len(graph.vertices)
    m = len(graph.edges)
    if n == 0:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion_diameter.empty_graph",
            message="graph must be nonempty",
        )
    if n > MAX_EDGE_DELETION_DIAMETER_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.edge_deletion_diameter.vertex_count",
            message=f"graph vertex count {n} exceeds {MAX_EDGE_DELETION_DIAMETER_VERTICES}",
        )
    if m > MAX_EDGE_DELETION_DIAMETER_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="graph.edge_deletion_diameter.edge_count",
            message=f"graph edge count {m} exceeds {MAX_EDGE_DELETION_DIAMETER_EDGES}",
        )
    # Check connected via NetworkX
    g = graph_from_value(graph)
    if not nx.is_connected(g):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.edge_deletion_diameter.not_connected",
            message="graph must be connected",
        )
    # Aggregate work bound: O(m*(n+m)) BFS; with n=64,m=256, that's ~80k operations, trivial
    # No further bound needed beyond vertex/edge caps for first envelope


def edge_deletion_diameter_profile(
    graph: SimpleUndirectedGraph,
) -> EdgeDeletionDiameterProfileResult:
    _admit(graph)
    g = graph_from_value(graph)
    source_diameter = int(nx.diameter(g))
    entries: list[EdgeDeletionDiameterEntry] = []
    for idx in range(len(graph.edges)):
        u, v = graph.edges[idx]
        h = g.copy()
        if h.has_edge(u, v):
            h.remove_edge(u, v)
        elif h.has_edge(v, u):
            h.remove_edge(v, u)
        else:
            raise OperationDomainValidationError(
                location=("graph", "edges", idx),
                code="graph.edge_deletion_diameter.edge_not_found",
                message=f"edge {(u, v)} not found in NetworkX graph",
            )
        edge: tuple[str, str] = graph.edges[idx]
        if not nx.is_connected(h):
            entries.append(
                EdgeDeletionDiameterEntry(
                    edge=edge,
                    edge_index=idx,
                    result="DISCONNECTED",
                    diameter=None,
                )
            )
        else:
            diam = int(nx.diameter(h))
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
