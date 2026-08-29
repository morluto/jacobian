"""Exact maximal-clique hypergraph construction."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from pydantic_core import PydanticCustomError

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphResult,
    _require_hypergraph_compatible_labels,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_maximal_clique_hypergraph"]


@dataclass(frozen=True)
class _CliqueEnumerationPlan:
    edges: tuple[tuple[str, tuple[str, ...]], ...]
    incidence_count: int
    result_wire_bytes: int


def _reject(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("graph",), code=code, message=message
    )


def _as_networkx_graph(graph: SimpleUndirectedGraph) -> nx.Graph[str]:
    backend: nx.Graph[str] = nx.Graph()
    backend.add_nodes_from(graph.vertices)
    backend.add_edges_from(graph.edges)
    return backend


def _enumeration_plan(graph: SimpleUndirectedGraph) -> _CliqueEnumerationPlan:
    """Enumerate once while enforcing the complete-family result ledger."""

    try:
        _require_hypergraph_compatible_labels(graph)
    except PydanticCustomError as error:
        raise _reject(error.type, str(error)) from error
    source_position = {vertex: index for index, vertex in enumerate(graph.vertices)}
    clique_members: list[tuple[str, ...]] = []
    incidence_count = 0

    # NetworkX 3.6 find_cliques is the maintained iterative Bron--Kerbosch /
    # Tomita kernel. It yields every maximal clique once, in unspecified order.
    for clique in nx.find_cliques(_as_networkx_graph(graph)):
        if len(clique) < 2:
            continue
        members = tuple(sorted(clique, key=source_position.__getitem__))
        clique_members.append(members)
        incidence_count += len(members)
        if len(clique_members) > MAX_EDGES:
            raise _reject(
                "graph.maximal_clique_hypergraph.edge_bound",
                "the complete maximal-clique family exceeds the "
                f"{MAX_EDGES:,}-edge hypergraph bound",
            )
        if incidence_count > MAX_TOTAL_INCIDENCES:
            raise _reject(
                "graph.maximal_clique_hypergraph.incidence_bound",
                "the complete maximal-clique family exceeds the "
                f"{MAX_TOTAL_INCIDENCES:,}-incidence hypergraph bound",
            )

    clique_members.sort(key=lambda members: tuple(source_position[v] for v in members))
    edges = tuple(
        (f"clique_{index}", members) for index, members in enumerate(clique_members)
    )
    result_payload = {
        "graph": graph.model_dump(mode="json"),
        "hypergraph": {
            "vertices": list(graph.vertices),
            "edges": [[edge_id, list(members)] for edge_id, members in edges],
        },
        "clique_count": len(edges),
    }
    try:
        result_wire_bytes = len(encode_strict_json(result_payload))
    except CanonicalizationError as error:
        raise _reject(
            "graph.maximal_clique_hypergraph.output_bound",
            "the source-bound maximal-clique hypergraph result exceeds the "
            f"{CanonicalLimits().max_output_bytes}-byte canonical output bound",
        ) from error
    output_limit = CanonicalLimits().max_output_bytes
    if result_wire_bytes > output_limit:
        raise _reject(
            "graph.maximal_clique_hypergraph.output_bound",
            "the source-bound maximal-clique hypergraph result exceeds the "
            f"{output_limit}-byte canonical output bound",
        )
    return _CliqueEnumerationPlan(
        edges=edges,
        incidence_count=incidence_count,
        result_wire_bytes=result_wire_bytes,
    )


def construct_maximal_clique_hypergraph(
    graph: SimpleUndirectedGraph,
) -> MaximalCliqueHypergraphResult:
    """Return every nontrivial inclusion-maximal clique as one hyperedge."""

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError(
            "construct_maximal_clique_hypergraph expects a SimpleUndirectedGraph"
        )
    plan = _enumeration_plan(graph)
    hypergraph = FiniteHypergraph(vertices=graph.vertices, edges=plan.edges)
    return MaximalCliqueHypergraphResult(
        graph=graph,
        hypergraph=hypergraph,
        clique_count=len(plan.edges),
    )
