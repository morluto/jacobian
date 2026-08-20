"""Typed wire contracts for chip-firing operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VERTICES = 50
MAX_DEGREE = 100


class LabelledGraph(StrictModel):
    """A finite undirected simple graph with labelled vertices."""

    vertices: tuple[str, ...] = Field(min_length=1, max_length=MAX_VERTICES)
    edges: tuple[tuple[str, str], ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_graph(self) -> Self:
        labels = set(self.vertices)
        if len(labels) != len(self.vertices):
            raise ValueError("vertex labels must be distinct")
        seen_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            u, v = edge
            if u not in labels or v not in labels:
                raise ValueError("every edge endpoint must be a declared vertex")
            if u == v:
                raise ValueError("self-loops are not allowed")
            canonical = (min(u, v), max(u, v))
            if canonical in seen_edges:
                raise ValueError("duplicate edges are not allowed in a simple graph")
            seen_edges.add(canonical)
        degree: dict[str, int] = dict.fromkeys(self.vertices, 0)
        for u, v in self.edges:
            degree[u] += 1
            degree[v] += 1
            if degree[u] > MAX_DEGREE or degree[v] > MAX_DEGREE:
                raise ValueError(f"vertex degree exceeds maximum {MAX_DEGREE}")
        if len(self.edges) > MAX_VERTICES * MAX_DEGREE // 2:
            raise ValueError("too many edges")
        return self


class LaplacianRequest(StrictModel):
    graph: LabelledGraph


class LaplacianResult(StrictModel):
    """The graph Laplacian matrix with degree vector."""

    vertices: tuple[str, ...]
    laplacian: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]


class FiringRequest(StrictModel):
    """Fire a vertex: transfer one chip to each neighbor."""

    graph: LabelledGraph
    divisor: tuple[int, ...] = Field(min_length=1)
    firing_vertex: str

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        if len(self.divisor) != len(self.graph.vertices):
            raise ValueError("divisor length must match vertex count")
        if self.firing_vertex not in set(self.graph.vertices):
            raise ValueError("firing vertex must be in the graph")
        return self


class FiringResult(StrictModel):
    """Result of firing a vertex."""

    vertex: str
    fired_divisor: tuple[int, ...]


class CriticalGroupRequest(StrictModel):
    """Request the critical group (sandpile group) of a graph."""

    graph: LabelledGraph
