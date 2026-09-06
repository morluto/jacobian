"""Typed wire contracts for quiver and path algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
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
    length: int = Field(ge=0, le=32)


# Results


class AdjacencyMatricesResult(StrictModel):
    """Two canonical matrix values on the retained vertex axis."""

    quiver: FiniteQuiver
    adjacency_matrix: IntegerMatrix
    transpose_matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_matrix_shapes(self) -> Self:
        n = self.quiver.vertex_count
        for matrix in (self.adjacency_matrix, self.transpose_matrix):
            if matrix.row_count != n or matrix.column_count != n:
                raise _validation_error(
                    "matrix_shape", "quiver matrices must be square on the vertex axis"
                )
        return self


class VertexProfilesResult(StrictModel):
    """In- and out-degree vectors on their shared implicit vertex axis."""

    in_degrees: tuple[int, ...]
    out_degrees: tuple[int, ...]


class FixedLengthPathsResult(StrictModel):
    quiver: FiniteQuiver
    length: int = Field(ge=0, le=32)
    path_matrix: IntegerMatrix
    total_paths: CanonicalInteger

    @model_validator(mode="after")
    def require_matrix_shape(self) -> Self:
        n = self.quiver.vertex_count
        if self.path_matrix.row_count != n or self.path_matrix.column_count != n:
            raise _validation_error(
                "path_matrix_shape", "path matrix must be square on the vertex axis"
            )
        return self
