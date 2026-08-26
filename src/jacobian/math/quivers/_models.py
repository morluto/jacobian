"""Typed wire contracts for quiver and path algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.quivers._path_bounds import fixed_length_paths_envelope

MAX_VERTICES = 128
MAX_ARROWS = 1024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"quiver.{reason}", message)


class FiniteQuiver(StrictModel):
    """A finite quiver (directed graph) with labelled vertices and arrows."""

    vertex_count: int = Field(ge=1, le=MAX_VERTICES)
    arrows: tuple[tuple[int, int], ...] = Field(default=(), max_length=MAX_ARROWS)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        for source, target in self.arrows:
            if not (0 <= source < self.vertex_count):
                raise _validation_error(
                    "arrow_source_out_of_range",
                    "arrow source must be in 0..vertex_count-1",
                )
            if not (0 <= target < self.vertex_count):
                raise _validation_error(
                    "arrow_target_out_of_range",
                    "arrow target must be in 0..vertex_count-1",
                )
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
        try:
            fixed_length_paths_envelope(
                vertex_count=self.quiver.vertex_count,
                arrow_count=len(self.quiver.arrows),
                length=self.length,
            )
        except ValueError as exc:
            raise _validation_error(
                "fixed_length_paths_exceeds_envelope", str(exc)
            ) from exc
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
