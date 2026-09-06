"""Typed wire contracts for algebraic topology operations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_EDGES = 64
MAX_WORD = 128


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(reason, message)


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


class EdgeGraph(StrictModel):
    """A bounded graph with an explicit vertex axis for edge paths."""

    vertex_count: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_EDGES)

    @classmethod
    def from_request(
        cls, vertex_count: int, edges: tuple[tuple[int, int], ...]
    ) -> EdgeGraph:
        return cls.model_construct(vertex_count=vertex_count, edges=edges)


class EdgePathWordResult(StrictModel):
    graph: EdgeGraph
    start_vertex: int
    path: tuple[OrientedEdge, ...] = Field(min_length=1, max_length=MAX_WORD)
    word: tuple[str, ...]
    length: int = Field(ge=0)


class EdgePathConcatenateResult(StrictModel):
    vertex_count: int = Field(ge=2)
    path_a: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)
    path_b: tuple[int, ...] = Field(min_length=2, max_length=MAX_WORD)
    path: tuple[int, ...]
    length: int = Field(ge=0)
