"""Typed wire contracts for algebraic topology operations."""

from __future__ import annotations

from typing import Self

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


class EdgePathWordRequest(StrictModel):
    """Compute the free group word for an edge path."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_EDGES)
    path: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
        if any(not 0 <= v < self.vertex_count for v in self.path):
            raise ValueError("path vertices must be in 0..vertex_count-1")
        edge_set = {frozenset((u, v)) for u, v in self.edges}
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            if frozenset((u, v)) not in edge_set:
                raise ValueError(f"path step {u}->{v} is not an edge in the graph")
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
