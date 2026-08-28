"""Typed contracts for the k-regular subgraph operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class RegularSubgraphRequest(StrictModel):
    """A simple graph and a target degree k for which a k-regular subgraph is sought."""

    graph: SimpleUndirectedGraph
    k: int = Field(ge=0)


class RegularSubgraphResult(StrictModel):
    """Result of a k-regular subgraph search."""

    graph: SimpleUndirectedGraph
    k: int
    found: bool
    vertices: tuple[str, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def require_empty_if_not_found(self) -> Self:
        if not self.found and (self.vertices or self.edges):
            raise PydanticCustomError(
                "graphs.regular_subgraph.witness_when_not_found",
                "vertices and edges must be empty when found is false",
            )
        return self


__all__ = [
    "RegularSubgraphRequest",
    "RegularSubgraphResult",
]
