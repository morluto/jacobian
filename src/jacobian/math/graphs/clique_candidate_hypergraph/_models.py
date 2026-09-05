"""Typed contracts for clique candidate hypergraphs."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


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
    members: tuple[str, ...] = Field(min_length=2)


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
    resource_map: tuple[ResourceEdgeMap, ...]
    candidate_map: tuple[CandidateCliqueMap, ...]
    candidate_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_maps_to_source(self) -> Self:
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


class AllCliqueCandidatesRequest(StrictModel):
    """Request every nontrivial clique as an edge-resource candidate family."""

    graph: SimpleUndirectedGraph


__all__ = [
    "AllCliqueCandidatesRequest",
    "CandidateCliqueMap",
    "CliqueCandidateHypergraphResult",
    "ResourceEdgeMap",
]
