"""Typed contracts for divisibility-incidence graph construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

MAX_FAMILY_SIZE: int = 256
MAX_TOTAL_FAMILY_SIZE: int = 256
MAX_GRAPH_EDGES: int = 32_640


class DivisibilityIncidenceGraphRequest(StrictModel):
    """Two finite positive-integer families whose divisibility incidence graph is constructed."""

    left_family: tuple[BoundedInteger, ...] = Field(
        max_length=MAX_FAMILY_SIZE,
        description="Unique positive integers labelling the left vertex family.",
    )
    right_family: tuple[BoundedInteger, ...] = Field(
        max_length=MAX_FAMILY_SIZE,
        description="Unique positive integers labelling the right vertex family.",
    )

    @model_validator(mode="after")
    def require_positive_families_and_graph_budget(self) -> Self:
        if any(
            parse_canonical_integer(value) <= 0
            for value in (*self.left_family, *self.right_family)
        ):
            raise _validation_error(
                "non_positive_family",
                "family values must be positive integers",
            )
        if len(set(self.left_family)) != len(self.left_family):
            raise _validation_error(
                "duplicate_left_family",
                "left_family values must be unique",
            )
        if len(set(self.right_family)) != len(self.right_family):
            raise _validation_error(
                "duplicate_right_family",
                "right_family values must be unique",
            )
        vertex_count = len(self.left_family) + len(self.right_family)
        if vertex_count > MAX_TOTAL_FAMILY_SIZE:
            raise _validation_error(
                "graph_vertex_budget",
                f"families must contain at most {MAX_TOTAL_FAMILY_SIZE} total values",
            )
        if len(self.left_family) * len(self.right_family) > MAX_GRAPH_EDGES:
            raise _validation_error(
                "graph_edge_budget",
                f"the incidence graph may contain at most {MAX_GRAPH_EDGES} edges",
            )
        return self


class DivisibilityIncidenceGraphResult(StrictModel):
    """Canonical bipartite simple graph with edges for each (l, r) with l | r."""

    left_family: tuple[BoundedInteger, ...]
    right_family: tuple[BoundedInteger, ...]
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_FAMILY_SIZE",
    "MAX_GRAPH_EDGES",
    "MAX_TOTAL_FAMILY_SIZE",
    "DivisibilityIncidenceGraphRequest",
    "DivisibilityIncidenceGraphResult",
]
