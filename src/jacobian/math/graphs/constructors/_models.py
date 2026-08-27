"""Typed contracts for graph constructor operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    IndexedSimpleUndirectedGraph,
    SimpleUndirectedGraph,
)

# ---------------------------------------------------------------------------
# Hypercube graph
# ---------------------------------------------------------------------------

MAX_HYPERCUBE_DIMENSION: int = 8  # 2^8 = 256 vertices, matching graph limit


class HypercubeGraphRequest(StrictModel):
    """One non-negative integer dimension d producing the d-dimensional hypercube Q_d."""

    dimension: StrictInt = Field(ge=0, le=MAX_HYPERCUBE_DIMENSION)


class HypercubeGraphResult(StrictModel):
    """The d-dimensional hypercube graph Q_d."""

    dimension: StrictInt
    graph: IndexedSimpleUndirectedGraph


# ---------------------------------------------------------------------------
# Keller graph
# ---------------------------------------------------------------------------

MAX_KELLER_DIMENSION: int = 4  # 4^4 = 256 vertices, matching graph limit


class KellerGraphRequest(StrictModel):
    """One non-negative integer dimension d producing the Keller graph K_d."""

    dimension: StrictInt = Field(ge=0, le=MAX_KELLER_DIMENSION)


class KellerGraphResult(StrictModel):
    """The Keller graph K_d over {0,1,2,3}^d words."""

    dimension: StrictInt
    graph: IndexedSimpleUndirectedGraph


# ---------------------------------------------------------------------------
# Triangle profile
# ---------------------------------------------------------------------------


class TriangleProfileRequest(StrictModel):
    """One finite simple undirected graph whose triangle profile is computed."""

    graph: SimpleUndirectedGraph


class TriangleProfileRow(StrictModel):
    """One triangle in a triangle profile."""

    vertices: tuple[str, str, str]


class TriangleProfileResult(StrictModel):
    """Complete triangle profile of a finite simple undirected graph."""

    source: SimpleUndirectedGraph
    triangles: tuple[TriangleProfileRow, ...]
    triangle_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def bind_triangle_count(self) -> Self:
        if self.triangle_count != len(self.triangles):
            raise PydanticCustomError(
                "graph.triangle_profile.count_mismatch",
                "triangle_count must equal the number of returned triangles",
            )
        return self


__all__ = [
    "MAX_HYPERCUBE_DIMENSION",
    "MAX_KELLER_DIMENSION",
    "HypercubeGraphRequest",
    "HypercubeGraphResult",
    "KellerGraphRequest",
    "KellerGraphResult",
    "TriangleProfileRequest",
    "TriangleProfileResult",
    "TriangleProfileRow",
]
