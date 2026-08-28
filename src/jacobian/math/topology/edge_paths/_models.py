"""Typed wire contracts for algebraic topology operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_EDGES = 64
MAX_WORD = 128


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(reason, message)


class EdgePath(StrictModel):
    """A path in a graph as a sequence of oriented edges."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise _validation_error(
                    "topology.edge_paths.require_valid_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
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


class EdgePathConcatenateRequest(StrictModel):
    """Concatenate two edge paths."""

    vertex_count: int = Field(ge=2)
    path_a: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)
    path_b: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)


# Results


class EdgePathWordResult(StrictModel):
    word: tuple[str, ...]
    length: int = Field(ge=0)


class EdgePathConcatenateResult(StrictModel):
    path: tuple[int, ...]
    length: int = Field(ge=0)
