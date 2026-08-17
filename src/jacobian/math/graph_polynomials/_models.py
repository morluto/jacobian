"""Typed wire contracts for exact graph polynomial operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class GraphEdge(StrictModel):
    """One undirected edge between two non-negative vertex indices."""

    u: int = Field(ge=0)
    v: int = Field(ge=0)

    @model_validator(mode="after")
    def require_no_loop(self) -> Self:
        if self.u == self.v:
            raise ValueError("graph edges must not be loops")
        return self


class GraphSpec(StrictModel):
    """A finite simple undirected graph as vertex count + edge list."""

    vertex_count: int = Field(ge=0, le=64)
    edges: tuple[GraphEdge, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        for edge in self.edges:
            if edge.u >= self.vertex_count or edge.v >= self.vertex_count:
                raise ValueError("edge endpoints must be < vertex_count")
        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            key = (min(edge.u, edge.v), max(edge.u, edge.v))
            if key in seen:
                raise ValueError("duplicate edges are not allowed")
            seen.add(key)
        return self


MAX_GRAPH_POLYNOMIAL_VERTICES = 12
MAX_GRAPH_POLYNOMIAL_EDGES = 24


class GraphPolynomialRequest(StrictModel):
    """Request a Tutte, chromatic, or flow polynomial on a tractable graph."""

    graph: GraphSpec

    @model_validator(mode="after")
    def require_deletion_contraction_budget(self) -> Self:
        if self.graph.vertex_count > MAX_GRAPH_POLYNOMIAL_VERTICES:
            raise ValueError(
                "Tutte, chromatic, and flow polynomials may have at most "
                f"{MAX_GRAPH_POLYNOMIAL_VERTICES} vertices"
            )
        if len(self.graph.edges) > MAX_GRAPH_POLYNOMIAL_EDGES:
            raise ValueError(
                "Tutte, chromatic, and flow polynomials may have at most "
                f"{MAX_GRAPH_POLYNOMIAL_EDGES} edges"
            )
        return self


MAX_MATCHING_VERTICES = 16
MAX_MATCHING_EDGES = 48


class MatchingPolynomialRequest(StrictModel):
    """Request a matching polynomial on a graph this recurrence can exhaust."""

    graph: GraphSpec

    @model_validator(mode="after")
    def require_matching_budget(self) -> Self:
        if self.graph.vertex_count > MAX_MATCHING_VERTICES:
            raise ValueError(
                f"matching polynomial graphs may have at most {MAX_MATCHING_VERTICES} vertices"
            )
        if len(self.graph.edges) > MAX_MATCHING_EDGES:
            raise ValueError(
                f"matching polynomial graphs may have at most {MAX_MATCHING_EDGES} edges"
            )
        return self


class PolynomialTerm(StrictModel):
    """One monomial term: coefficient times x^degree."""

    coefficient: int
    degree: int = Field(ge=0)


class GraphPolynomialResult(StrictModel):
    """A sparse polynomial represented as a list of (coefficient, degree) terms."""

    terms: tuple[PolynomialTerm, ...]

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        degrees = [term.degree for term in self.terms]
        if degrees != sorted(degrees):
            raise ValueError("polynomial terms must be sorted by degree")
        if len(set(degrees)) != len(degrees):
            raise ValueError("polynomial degrees must be unique")
        if any(term.coefficient == 0 for term in self.terms):
            raise ValueError("polynomial terms must have nonzero coefficients")
        return self


__all__ = [
    "GraphEdge",
    "GraphPolynomialRequest",
    "GraphPolynomialResult",
    "GraphSpec",
    "MatchingPolynomialRequest",
    "PolynomialTerm",
]
