"""Typed wire contracts for quiver and path algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.matrices.values import IntegerMatrix

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
    length: int = Field(strict=True, ge=0, le=32)


# Results


class AdjacencyMatricesResult(StrictModel):
    """Two square matrix values on the retained quiver vertex axis."""

    quiver: FiniteQuiver
    adjacency_matrix: IntegerMatrix
    transpose_matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_vertex_axes(self) -> Self:
        size = self.quiver.vertex_count
        if any(
            matrix.row_count != size or matrix.column_count != size
            for matrix in (self.adjacency_matrix, self.transpose_matrix)
        ):
            raise _validation_error(
                "adjacency_axes", "adjacency matrices must use both quiver vertex axes"
            )
        return self


class VertexProfilesResult(StrictModel):
    """In- and out-degree vectors on the retained quiver vertex axis."""

    quiver: FiniteQuiver
    in_degrees: tuple[int, ...]
    out_degrees: tuple[int, ...]

    @model_validator(mode="after")
    def require_vertex_axes(self) -> Self:
        if (
            len(self.in_degrees) != self.quiver.vertex_count
            or len(self.out_degrees) != self.quiver.vertex_count
        ):
            raise _validation_error(
                "profile_axes", "degree profiles must use the quiver vertex axis"
            )
        return self


class FixedLengthPathsResult(StrictModel):
    """Exact path counts on the retained quiver axes at one declared length."""

    quiver: FiniteQuiver
    length: int = Field(strict=True, ge=0, le=32)
    path_matrix: IntegerMatrix
    total_paths: ExactInteger

    @model_validator(mode="after")
    def require_vertex_axes(self) -> Self:
        size = self.quiver.vertex_count
        if self.path_matrix.row_count != size or self.path_matrix.column_count != size:
            raise _validation_error(
                "path_axes", "path matrix must use both quiver vertex axes"
            )
        return self
