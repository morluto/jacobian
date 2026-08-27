"""Typed contracts for fixed-length simple path profiles."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class PathProfileRequest(StrictModel):
    """A finite simple undirected graph and a path length."""

    graph: SimpleUndirectedGraph
    path_length: int = Field(ge=0, le=10)


class PathProfileRow(StrictModel):
    """One (source, target, count) triple in a path profile."""

    source: str
    target: str
    path_count: int = Field(ge=0)


class PathProfileResult(StrictModel):
    """Complete fixed-length simple path profile by endpoint."""

    source: SimpleUndirectedGraph
    path_length: int
    rows: list[PathProfileRow]


__all__ = [
    "PathProfileRequest",
    "PathProfileResult",
    "PathProfileRow",
]
