"""Typed contracts for the maximal-clique hypergraph operation."""

from __future__ import annotations

from typing import Self

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_LABEL_LENGTH,
    FiniteHypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _require_hypergraph_compatible_labels(graph: SimpleUndirectedGraph) -> None:
    for label in graph.vertices:
        if len(label) > MAX_LABEL_LENGTH:
            raise PydanticCustomError(
                "graph.maximal_clique_hypergraph.label_length",
                "graph vertex labels must fit the hypergraph label bound of "
                f"{MAX_LABEL_LENGTH} characters",
            )
        try:
            label.encode("utf-8")
        except UnicodeEncodeError as error:
            raise PydanticCustomError(
                "graph.maximal_clique_hypergraph.label_encoding",
                "graph vertex labels must be valid UTF-8",
            ) from error


class MaximalCliqueHypergraphRequest(StrictModel):
    """Request to construct the maximal-clique hypergraph of a graph."""

    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_hypergraph_compatible_labels(self) -> Self:
        _require_hypergraph_compatible_labels(self.graph)
        return self


class MaximalCliqueHypergraphResult(StrictModel):
    """The maximal-clique hypergraph of a graph."""

    graph: SimpleUndirectedGraph
    hypergraph: FiniteHypergraph
    clique_count: StrictInt

    @model_validator(mode="after")
    def bind_source_and_count(self) -> Self:
        if self.hypergraph.vertices != self.graph.vertices:
            raise PydanticCustomError(
                "graph.maximal_clique_hypergraph.source_vertices",
                "hypergraph vertices must equal the source graph vertices",
            )
        if self.clique_count != len(self.hypergraph.edges):
            raise PydanticCustomError(
                "graph.maximal_clique_hypergraph.clique_count",
                "clique_count must equal the number of hypergraph edges",
            )
        return self


__all__ = [
    "MaximalCliqueHypergraphRequest",
    "MaximalCliqueHypergraphResult",
]
