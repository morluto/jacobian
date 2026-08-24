"""Typed wire contracts for quiver and path algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VERTICES = 128
MAX_ARROWS = 1024


class FiniteQuiver(StrictModel):
    """A finite quiver (directed graph) with labelled vertices and arrows."""

    vertex_count: int = Field(ge=1, le=MAX_VERTICES)
    arrows: tuple[tuple[int, int], ...] = Field(default=(), max_length=MAX_ARROWS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for source, target in self.arrows:
            if not (0 <= source < self.vertex_count):
                raise ValueError("arrow source must be in 0..vertex_count-1")
            if not (0 <= target < self.vertex_count):
                raise ValueError("arrow target must be in 0..vertex_count-1")
        return self


class AdjacencyMatricesRequest(StrictModel):
    quiver: FiniteQuiver


class VertexProfilesRequest(StrictModel):
    quiver: FiniteQuiver


class FixedLengthPathsRequest(StrictModel):
    quiver: FiniteQuiver
    # Path length is decoupled from the vertex bound; keep a
    # conservative 32-step cap and bound work via explicit work budget
    # rather than tying length to MAX_VERTICES.
    length: int = Field(ge=0, le=32)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        return self


# Results


class AdjacencyMatricesResult(StrictModel):
    adjacency_matrix: tuple[tuple[int, ...], ...]
    transpose_matrix: tuple[tuple[int, ...], ...]
    vertex_count: int = Field(ge=1)
    method: str = "ADjaCENCY_CONSTRUCTION"


class VertexProfilesResult(StrictModel):
    in_degrees: tuple[int, ...]
    out_degrees: tuple[int, ...]
    vertex_count: int = Field(ge=1)
    method: str = "DEGREE_COUNT"


class FixedLengthPathsResult(StrictModel):
    path_matrix: tuple[tuple[int, ...], ...]
    total_paths: int = Field(ge=0)
    method: str = "MATRIX_POWER"
