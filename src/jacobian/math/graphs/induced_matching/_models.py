"""Typed contracts for maximum induced matching."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.graphs.independence import (
    IndependenceNumberBudget,
    IndependenceSearchStatus,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


class MaximumInducedMatchingRequest(StrictModel):
    graph: SimpleUndirectedGraph
    resource_budget: IndependenceNumberBudget = Field(
        default_factory=IndependenceNumberBudget
    )


class SourceEdgeBinding(StrictModel):
    edge_id: str
    endpoints: tuple[str, str]


class MaximumInducedMatchingResult(StrictModel):
    graph: SimpleUndirectedGraph
    source_edges: tuple[SourceEdgeBinding, ...]
    status: IndependenceSearchStatus
    cardinality: StrictInt = Field(ge=0, le=128)
    lower_bound: StrictInt = Field(ge=0, le=128)
    upper_bound: StrictInt = Field(ge=0, le=128)
    selected_edge_ids: tuple[str, ...] = Field(max_length=128)
    induced_endpoint_graph: SimpleUndirectedGraph
