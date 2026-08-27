"""Typed contracts for graph constructor operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, PrivateAttr, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.constructors._bounds import (
    TriangleProfileAdmission,
    admit_triangle_profile,
)
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
    _admission: TriangleProfileAdmission | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def require_bounded_triangle_profile(self) -> Self:
        self._admission = admit_triangle_profile(self.graph)
        return self

    def admitted_profile(self) -> TriangleProfileAdmission:
        """Return the request's cached exact scan plan."""

        admission = self._admission
        if admission is None:
            admission = admit_triangle_profile(self.graph)
            self._admission = admission
        return admission


class TriangleProfileRow(StrictModel):
    """One triangle in a triangle profile."""

    vertices: tuple[str, str, str]


class TriangleProfileResult(StrictModel):
    """Complete triangle profile of a finite simple undirected graph."""

    source: SimpleUndirectedGraph
    triangles: tuple[TriangleProfileRow, ...]
    triangle_count: StrictInt


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
