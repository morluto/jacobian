"""Typed contracts for divisibility-incidence graph construction."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_FAMILY_SIZE: int = 256


class DivisibilityIncidenceGraphRequest(StrictModel):
    """Two finite positive-integer families whose divisibility incidence graph is constructed."""

    left_family: list[str] = Field(max_length=MAX_FAMILY_SIZE)
    right_family: list[str] = Field(max_length=MAX_FAMILY_SIZE)


class DivisibilityIncidenceGraphResult(StrictModel):
    """Canonical bipartite simple graph with edges for each (l, r) with l | r."""

    left_family: list[str]
    right_family: list[str]
    graph: SimpleUndirectedGraph


__all__ = [
    "DivisibilityIncidenceGraphRequest",
    "DivisibilityIncidenceGraphResult",
    "MAX_FAMILY_SIZE",
]
