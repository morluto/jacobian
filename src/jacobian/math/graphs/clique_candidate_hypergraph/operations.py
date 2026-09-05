"""Clique candidate hypergraphs over graph-edge resources."""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.clique_candidate_hypergraph._models import (
    CandidateCliqueMap,
    CliqueCandidateHypergraphResult,
    ResourceEdgeMap,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "construct_all_clique_candidate_hypergraph",
    "convert_candidate_cliques",
]


def _reject(code: str, message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("graph",), code=code, message=message
    )


def _resource_plan(
    graph: SimpleUndirectedGraph,
) -> tuple[tuple[ResourceEdgeMap, ...], dict[tuple[str, str], str]]:
    """Assign one zero-padded resource ID per canonical source edge."""

    ordered_edges = tuple(
        (first, second)
        for first, second in (tuple(sorted(edge)) for edge in graph.edges)
    )
    width = len(str(len(ordered_edges)))
    resource_map = tuple(
        ResourceEdgeMap(resource=f"e{index:0{width}d}", endpoints=edge)
        for index, edge in enumerate(ordered_edges)
    )
    resource_of = {entry.endpoints: entry.resource for entry in resource_map}
    return resource_map, resource_of


def _candidate_hypergraph(
    graph: SimpleUndirectedGraph,
    resource_map: tuple[ResourceEdgeMap, ...],
    resource_of: dict[tuple[str, str], str],
    candidates: tuple[tuple[str, ...], ...],
) -> CliqueCandidateHypergraphResult:
    """Build the result value for already-validated complete candidates."""

    width = len(str(len(candidates)))
    candidate_ids = tuple(f"q{index:0{width}d}" for index in range(len(candidates)))
    hyperedges = tuple(
        (
            candidate_id,
            tuple(sorted(resource_of[_ordered(edge)] for edge in _pairs(members))),
        )
        for candidate_id, members in zip(candidate_ids, candidates, strict=True)
    )
    hypergraph = FiniteHypergraph(
        vertices=tuple(entry.resource for entry in resource_map),
        edges=hyperedges,
    )
    return CliqueCandidateHypergraphResult._from_kernel(
        graph=graph,
        hypergraph=hypergraph,
        resource_map=resource_map,
        candidate_map=tuple(
            CandidateCliqueMap(candidate=candidate_id, members=members)
            for candidate_id, members in zip(candidate_ids, candidates, strict=True)
        ),
    )


def _ordered(edge: tuple[str, str]) -> tuple[str, str]:
    left, right = edge
    return (left, right) if left < right else (right, left)


def _pairs(members: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    for left_index in range(len(members)):
        for right_index in range(left_index + 1, len(members)):
            yield (members[left_index], members[right_index])


def _adjacency(graph: SimpleUndirectedGraph) -> dict[str, frozenset[str]]:
    neighbors: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for left, right in graph.edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return {vertex: frozenset(peers) for vertex, peers in neighbors.items()}


def convert_candidate_cliques(
    graph: SimpleUndirectedGraph,
    candidates: tuple[tuple[str, ...], ...],
) -> CliqueCandidateHypergraphResult:
    """Convert a supplied candidate-clique family to edge resources.

    Native adapter: each candidate must be complete with at least two
    vertices, else a typed domain error names its position. Duplicate
    member sets keep distinct candidate IDs. Conversion work is quadratic
    in candidate sizes; complete enumeration is the separate public
    constructor below.
    """

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("convert_candidate_cliques expects a SimpleUndirectedGraph")
    carrier = set(graph.vertices)
    adjacency = _adjacency(graph)
    validated: list[tuple[str, ...]] = []
    for index, candidate in enumerate(candidates):
        members = tuple(candidate)
        if len(members) < 2:
            raise OperationDomainValidationError(
                location=("candidates", index),
                code="graph.clique_candidate.candidate_too_small",
                message=f"candidate {index} must hold at least two vertices",
            )
        if len(set(members)) != len(members):
            raise OperationDomainValidationError(
                location=("candidates", index),
                code="graph.clique_candidate.candidate_members_not_unique",
                message=f"candidate {index} must hold distinct vertices",
            )
        unknown = set(members) - carrier
        if unknown:
            raise OperationDomainValidationError(
                location=("candidates", index),
                code="graph.clique_candidate.candidate_vertex_unknown",
                message=f"candidate {index} must use declared graph vertices",
            )
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                left, right = members[left_position], members[right_position]
                if right not in adjacency[left]:
                    first, second = (left, right) if left < right else (right, left)
                    raise OperationDomainValidationError(
                        location=("candidates", index),
                        code="graph.clique_candidate.candidate_not_complete",
                        message=(
                            f"candidate {index} is not complete: "
                            f"{first}-{second} is not a graph edge"
                        ),
                    )
        validated.append(members)
    resource_map, resource_of = _resource_plan(graph)
    return _candidate_hypergraph(graph, resource_map, resource_of, tuple(validated))


def construct_all_clique_candidate_hypergraph(
    graph: SimpleUndirectedGraph,
) -> CliqueCandidateHypergraphResult:
    """Return every nontrivial clique as an edge-resource candidate family.

    Maximal cliques come from the maintained NetworkX Bron-Kerbosch kernel;
    every subset of order at least two is a candidate, deduplicated across
    maximal cliques. Output is bounded incrementally by the hypergraph edge
    and incidence envelopes before materialization.
    """

    from itertools import combinations

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError(
            "construct_all_clique_candidate_hypergraph expects a SimpleUndirectedGraph"
        )
    resource_map, resource_of = _resource_plan(graph)
    backend: nx.Graph[str] = nx.Graph()
    backend.add_nodes_from(graph.vertices)
    backend.add_edges_from(graph.edges)
    seen: set[tuple[str, ...]] = set()
    cliques: list[tuple[str, ...]] = []
    incidences = 0
    for maximal in nx.find_cliques(backend):
        if len(maximal) < 2:
            continue
        ordered = tuple(sorted(maximal))
        for width in range(2, len(ordered) + 1):
            for combo in combinations(ordered, width):
                if combo in seen:
                    continue
                seen.add(combo)
                cliques.append(combo)
                incidences += width * (width - 1) // 2
                if len(cliques) > MAX_EDGES:
                    raise _reject(
                        "graph.clique_candidate.edge_bound",
                        "the complete clique-candidate family exceeds the "
                        f"{MAX_EDGES:,}-edge hypergraph bound",
                    )
                if incidences > MAX_TOTAL_INCIDENCES:
                    raise _reject(
                        "graph.clique_candidate.incidence_bound",
                        "the complete clique-candidate family exceeds the "
                        f"{MAX_TOTAL_INCIDENCES:,}-incidence hypergraph bound",
                    )
    source_position = {vertex: index for index, vertex in enumerate(graph.vertices)}
    cliques.sort(key=lambda members: tuple(source_position[v] for v in members))
    return _candidate_hypergraph(graph, resource_map, resource_of, tuple(cliques))
