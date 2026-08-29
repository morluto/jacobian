"""Typed contracts for the monochromatic clique hypergraph operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_VERTICES = 20
MAX_CLIQUE_ORDER = 8


class MonochromaticCliqueHypergraphRequest(StrictModel):
    """Request to construct the monochromatic clique hypergraph."""

    colored_graph: ColoredUndirectedGraph
    clique_order: int = Field(ge=2, le=MAX_CLIQUE_ORDER)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.colored_graph.edge_colors:
            raise PydanticCustomError(
                "monochromatic_clique.no_edge_colors",
                "edge_colors must be provided (total colouring required)",
            )
        if len(self.colored_graph.graph.vertices) > MAX_VERTICES:
            raise PydanticCustomError(
                "monochromatic_clique.too_many_vertices",
                f"at most {MAX_VERTICES} vertices are supported",
            )
        vertices = self.colored_graph.graph.vertices
        n = len(vertices)
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    vertices[i],
                    vertices[j],
                ) not in self.colored_graph.graph.edges and (
                    vertices[j],
                    vertices[i],
                ) not in self.colored_graph.graph.edges:
                    raise PydanticCustomError(
                        "monochromatic_clique.graph_not_complete",
                        "the underlying graph must be complete",
                    )
        return self


class MonochromaticCliqueHypergraphResult(StrictModel):
    """The monochromatic clique hypergraph of a coloured complete graph."""

    colored_graph: ColoredUndirectedGraph
    clique_order: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_CLIQUE_ORDER",
    "MAX_VERTICES",
    "MonochromaticCliqueHypergraphRequest",
    "MonochromaticCliqueHypergraphResult",
]
