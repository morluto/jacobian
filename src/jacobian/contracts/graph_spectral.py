"""Typed wire contracts for graph spectral operations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from jacobian.contracts.base import ContractModel


class GraphEdgeList(ContractModel):
    """A simple undirected graph given by an edge list."""

    vertex_count: int = Field(ge=1, le=32)
    edges: tuple[tuple[int, int], ...] = Field(max_length=512)


class GraphSpectrumRequest(ContractModel):
    graph: GraphEdgeList


class GraphSpectrumResult(ContractModel):
    """The exact eigenvalues with algebraic multiplicities of a graph matrix."""

    eigenvalues: tuple[str, ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"
