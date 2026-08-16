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


class GraphCharacteristicPolynomialRequest(ContractModel):
    """Request the characteristic polynomial of a graph matrix."""

    graph: GraphEdgeList
    matrix: Literal["ADJACENCY", "LAPLACIAN"] = "ADJACENCY"


class GraphCharacteristicPolynomialResult(ContractModel):
    """Dense monic characteristic polynomial coefficients of a graph matrix."""

    variable: Literal["lambda"] = "lambda"
    degree: int = Field(ge=0, le=32)
    coefficients_descending: tuple[str, ...] = Field(min_length=1, max_length=33)
    monic: Literal[True] = True
    matrix: Literal["ADJACENCY", "LAPLACIAN"]
    convention: Literal["DET_LAMBDA_I_MINUS_M"] = "DET_LAMBDA_I_MINUS_M"

    @model_validator(mode="after")
    def require_dense_monic_coefficients(self) -> Self:
        if len(self.coefficients_descending) != self.degree + 1:
            raise ValueError("dense coefficient count must be degree plus one")
        if self.coefficients_descending[0] != "1":
            raise ValueError("characteristic polynomial must be monic")
        return self
