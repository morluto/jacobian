"""Typed contracts for clique candidate hypergraphs."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_EDGES,
    MAX_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class ResourceEdgeMap(StrictModel):
    """One edge-resource ID bound to its source graph edge endpoints."""

    resource: str = Field(min_length=1)
    endpoints: tuple[str, str]

    @model_validator(mode="after")
    def require_ordered_endpoints(self) -> Self:
        if self.endpoints[0] >= self.endpoints[1]:
            raise _validation_error(
                "graph.clique_candidate.resource_order",
                "resource endpoints must use lexicographic label order",
            )
        return self


class CandidateCliqueMap(StrictModel):
    """One candidate ID bound to its original vertex subset."""

    candidate: str = Field(min_length=1)
    members: tuple[str, ...] = Field(min_length=2, max_length=MAX_SIMPLE_GRAPH_VERTICES)


class CliqueCandidateHypergraphResult(StrictModel):
    """Candidate cliques as hyperedges over graph-edge resources.

    ``hypergraph`` has one vertex per source graph edge and one hyperedge
    per candidate whose members are exactly the candidate's internal edges.
    ``resource_map`` binds each resource ID to its source endpoints and
    ``candidate_map`` binds each candidate ID to its vertex subset, so both
    transports survive serialization and relabeling structurally.
    """

    graph: SimpleUndirectedGraph
    hypergraph: FiniteHypergraph
    resource_map: tuple[ResourceEdgeMap, ...] = Field(max_length=MAX_SIMPLE_GRAPH_EDGES)
    candidate_map: tuple[CandidateCliqueMap, ...] = Field(max_length=MAX_EDGES)
    candidate_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_maps_to_source(self) -> Self:
        if any(
            len(entry.members) > MAX_SIMPLE_GRAPH_VERTICES
            for entry in self.candidate_map
        ):
            raise _validation_error(
                "graph.clique_candidate.candidate_size",
                "candidate members cannot exceed the graph vertex bound",
            )
        pair_work = sum(
            len(entry.members) * (len(entry.members) - 1) // 2
            for entry in self.candidate_map
        )
        if pair_work > MAX_TOTAL_INCIDENCES:
            raise _validation_error(
                "graph.clique_candidate.incidence_bound",
                "candidate internal-edge supports exceed the incidence bound",
            )
        resources = {entry.resource: entry.endpoints for entry in self.resource_map}
        if len(resources) != len(self.resource_map):
            raise _validation_error(
                "graph.clique_candidate.resource_identity",
                "resource IDs must be unique",
            )
        if tuple(entry.resource for entry in self.resource_map) != tuple(
            sorted(resources)
        ):
            raise _validation_error(
                "graph.clique_candidate.resource_order",
                "resource map entries must use canonical resource order",
            )
        source_edges = {tuple(sorted(edge)) for edge in self.graph.edges}
        if set(resources.values()) != source_edges:
            raise _validation_error(
                "graph.clique_candidate.resource_coverage",
                "resource map must cover exactly the source graph edges",
            )
        if set(self.hypergraph.vertices) != set(resources):
            raise _validation_error(
                "graph.clique_candidate.hypergraph_resource_coverage",
                "hypergraph vertices must be exactly the resource IDs",
            )
        if tuple(self.hypergraph.vertices) != tuple(
            entry.resource for entry in self.resource_map
        ):
            raise _validation_error(
                "graph.clique_candidate.hypergraph_resource_order",
                "hypergraph vertices must use canonical resource order",
            )
        candidates = [entry.candidate for entry in self.candidate_map]
        if len(set(candidates)) != len(candidates):
            raise _validation_error(
                "graph.clique_candidate.candidate_identity",
                "candidate IDs must be unique",
            )
        if self.candidate_count != len(self.candidate_map):
            raise _validation_error(
                "graph.clique_candidate.candidate_count",
                "candidate_count must equal the number of candidate entries",
            )
        if tuple(entry.candidate for entry in self.candidate_map) != tuple(
            sorted(candidates)
        ):
            raise _validation_error(
                "graph.clique_candidate.candidate_order",
                "candidate map entries must use canonical candidate order",
            )
        _require_candidate_bindings(
            self.graph,
            self.hypergraph,
            self.resource_map,
            self.candidate_map,
            candidates,
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        hypergraph: FiniteHypergraph,
        resource_map: tuple[ResourceEdgeMap, ...],
        candidate_map: tuple[CandidateCliqueMap, ...],
    ) -> Self:
        """Construct a result whose resource transports the kernel verified."""

        return cls.model_construct(
            graph=graph,
            hypergraph=hypergraph,
            resource_map=resource_map,
            candidate_map=candidate_map,
            candidate_count=len(candidate_map),
        )


def _require_candidate_bindings(
    graph: SimpleUndirectedGraph,
    hypergraph: FiniteHypergraph,
    resource_map: tuple[ResourceEdgeMap, ...],
    candidate_map: tuple[CandidateCliqueMap, ...],
    candidates: list[str],
) -> None:
    resources = {entry.resource: entry.endpoints for entry in resource_map}
    resource_for_edge = {
        endpoints: resource for resource, endpoints in resources.items()
    }
    hyperedges = dict(hypergraph.edges)
    if set(hyperedges) != set(candidates):
        raise _validation_error(
            "graph.clique_candidate.hypergraph_candidate_coverage",
            "hypergraph edges must be exactly the candidate IDs",
        )
    for entry in candidate_map:
        if len(set(entry.members)) != len(entry.members):
            raise _validation_error(
                "graph.clique_candidate.candidate_identity",
                "candidate members must be distinct",
            )
        if not set(entry.members) <= set(graph.vertices):
            raise _validation_error(
                "graph.clique_candidate.candidate_vertex_unknown",
                "candidate members must use declared graph vertices",
            )
        try:
            expected_resources = tuple(
                sorted(
                    resource_for_edge[(left, right) if left < right else (right, left)]
                    for index, left in enumerate(entry.members)
                    for right in entry.members[index + 1 :]
                )
            )
        except KeyError as error:
            raise _validation_error(
                "graph.clique_candidate.candidate_not_complete",
                "candidate members must form a clique in the source graph",
            ) from error
        if (
            len(expected_resources)
            != len(entry.members) * (len(entry.members) - 1) // 2
        ):
            raise _validation_error(
                "graph.clique_candidate.candidate_not_complete",
                "candidate members must form a clique in the source graph",
            )
        if hyperedges[entry.candidate] != expected_resources:
            raise _validation_error(
                "graph.clique_candidate.hypergraph_candidate_binding",
                "hypergraph edge resources must match candidate clique members",
            )


class AllCliqueCandidatesRequest(StrictModel):
    """Request every nontrivial clique as an edge-resource candidate family."""

    graph: SimpleUndirectedGraph


__all__ = [
    "AllCliqueCandidatesRequest",
    "CandidateCliqueMap",
    "CliqueCandidateHypergraphResult",
    "ResourceEdgeMap",
]
