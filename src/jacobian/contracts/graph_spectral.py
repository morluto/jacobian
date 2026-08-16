"""Typed wire contracts for graph spectral operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class GraphEdgeList(ContractModel):
    """A simple undirected graph given by an edge list."""

    vertex_count: int = Field(ge=1, le=32)
    edges: tuple[tuple[int, int], ...] = Field(max_length=512)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if u == v:
                raise ValueError("a simple graph cannot contain self-loops")
            edge = (min(u, v), max(u, v))
            if edge in seen:
                raise ValueError("a simple graph cannot contain duplicate edges")
            seen.add(edge)
        return self


class GraphSpectrumRequest(ContractModel):
    graph: GraphEdgeList


class GraphSpectrumResult(ContractModel):
    """The exact eigenvalues with algebraic multiplicities of a graph matrix."""

    eigenvalues: tuple[str, ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"
