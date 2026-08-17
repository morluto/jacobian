"""Provider-independent values for finite simple undirected graphs."""

from __future__ import annotations

import unicodedata
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

GraphCompositionOperation = Literal[
    "DISJOINT_UNION",
    "JOIN",
    "COMPLEMENT",
    "LEXICOGRAPHIC_PRODUCT",
]


class SimpleUndirectedGraph(StrictModel):
    """Immutable canonical value for a finite simple undirected graph."""

    graph_schema_version: Literal["1"] = "1"
    vertices: tuple[str, ...] = Field(max_length=256)
    edges: tuple[tuple[str, str], ...] = Field(max_length=32640)

    @model_validator(mode="after")
    def require_canonical_simple_graph(self) -> Self:
        if any(
            not unicodedata.is_normalized("NFC", vertex) for vertex in self.vertices
        ):
            raise ValueError("graph vertices must use Unicode NFC")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        if any(
            left >= right or left not in self.vertices or right not in self.vertices
            for left, right in self.edges
        ):
            raise ValueError("edges must contain two declared vertices in order")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("graph edges must be unique")
        return self


class GraphCompositionInput(StrictModel):
    """Two exact graphs and one explicit graph composition operation."""

    operation: GraphCompositionOperation
    left: SimpleUndirectedGraph
    right: SimpleUndirectedGraph | None = None

    @model_validator(mode="after")
    def require_operands(self) -> Self:
        if self.operation == "COMPLEMENT":
            if self.right is not None:
                raise ValueError("complement does not accept a right graph")
        elif self.right is None:
            raise ValueError(f"{self.operation} requires a right graph")
        return self


__all__ = [
    "GraphCompositionInput",
    "GraphCompositionOperation",
    "SimpleUndirectedGraph",
]
