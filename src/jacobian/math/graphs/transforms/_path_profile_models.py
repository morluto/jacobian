"""Typed contracts for fixed-length simple path profiles."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_PATH_PROFILE_SEARCH_WORK = 10_000_000
MAX_PATH_PROFILE_ORDER = 256
_MAX_PATH_PROFILE_ROWS = MAX_PATH_PROFILE_ORDER * MAX_PATH_PROFILE_ORDER


def _canonical_max_degree(graph: SimpleUndirectedGraph) -> int:
    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    return max(degrees.values(), default=0)


def _path_prefix_work_bound(vertex_count: int, max_degree: int, length: int) -> int:
    """Bound DFS nodes for one source by degree and simple-path length."""
    depth = min(length, max(vertex_count - 1, 0))
    prefixes = 1
    total = 1
    for step in range(1, depth + 1):
        prefixes *= max_degree if step == 1 else max(max_degree - 1, 0)
        total += prefixes
    return total


class PathProfileRequest(StrictModel):
    """A finite simple undirected graph and a path length."""

    graph: SimpleUndirectedGraph
    path_length: int = Field(ge=0, le=10)

    @model_validator(mode="after")
    def require_row_envelope(self) -> Self:
        order = len(self.graph.vertices)
        if order > MAX_PATH_PROFILE_ORDER:
            raise PydanticCustomError(
                "graph.path_profile.row_bound",
                "path-profile computation supports at most "
                f"{MAX_PATH_PROFILE_ORDER} vertices",
            )
        return self


class PathProfileRow(StrictModel):
    """One (source, target, count) triple in a path profile."""

    source: str
    target: str
    path_count: int = Field(ge=0)


class PathProfileResult(StrictModel):
    """Complete fixed-length simple path profile by endpoint."""

    source: SimpleUndirectedGraph
    path_length: int = Field(ge=0, le=10)
    rows: list[PathProfileRow] = Field(max_length=_MAX_PATH_PROFILE_ROWS)


__all__ = [
    "MAX_PATH_PROFILE_ORDER",
    "MAX_PATH_PROFILE_SEARCH_WORK",
    "PathProfileRequest",
    "PathProfileResult",
    "PathProfileRow",
]
