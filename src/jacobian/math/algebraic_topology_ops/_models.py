"""Typed wire contracts for algebraic topology operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_EDGES = 64
MAX_WORD = 128


class EdgePath(StrictModel):
    """A path in a graph as a sequence of oriented edges."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
        return self


class OrientedEdge(StrictModel):
    edge_index: int = Field(ge=0)
    orientation: Literal[-1, 1]


class EdgePathWordRequest(StrictModel):
    """Compute the free group word for an edge path."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_EDGES)
    start_vertex: int = Field(ge=0)
    path: tuple[OrientedEdge, ...] = Field(min_length=1, max_length=MAX_WORD)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
        if self.start_vertex >= self.vertex_count:
            raise ValueError("start vertex must be in 0..vertex_count-1")
        current = self.start_vertex
        for step in self.path:
            if step.edge_index >= len(self.edges):
                raise ValueError("path edge index is outside the graph")
            left, right = self.edges[step.edge_index]
            source, target = (left, right) if step.orientation == 1 else (right, left)
            if source != current:
                raise ValueError("oriented edge path is not continuous")
            current = target
        return self


class EdgePathConcatenateRequest(StrictModel):
    """Concatenate two edge paths."""

    vertex_count: int = Field(ge=2)
    path_a: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)
    path_b: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if any(not 0 <= v < self.vertex_count for v in self.path_a):
            raise ValueError("path_a vertices must be valid")
        if any(not 0 <= v < self.vertex_count for v in self.path_b):
            raise ValueError("path_b vertices must be valid")
        if self.path_a[-1] != self.path_b[0]:
            raise ValueError(
                f"last vertex of path_a ({self.path_a[-1]}) must equal "
                f"first vertex of path_b ({self.path_b[0]})"
            )
        return self


# Results


class EdgePathWordResult(StrictModel):
    word: tuple[str, ...]
    length: int = Field(ge=0)
    method: str = "EDGE_LABEL_REDUCTION"


class EdgePathConcatenateResult(StrictModel):
    path: tuple[int, ...]
    length: int = Field(ge=0)
    method: str = "PATH_CONCATENATION"
