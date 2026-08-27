"""Typed contracts for divisibility-incidence graph construction."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS, _validation_error

MAX_FAMILY_SIZE: int = 256
MAX_TOTAL_FAMILY_SIZE: int = 256
MAX_GRAPH_EDGES: int = 32_640

PositiveInteger = Annotated[
    str,
    StringConstraints(
        pattern=rf"^[1-9][0-9]{{0,{MAX_INTEGER_DIGITS - 1}}}$",
        max_length=MAX_INTEGER_DIGITS,
        strict=True,
    ),
]


class DivisibilityIncidenceGraphRequest(StrictModel):
    """Two finite positive-integer families whose divisibility incidence graph is constructed."""

    left_family: list[PositiveInteger] = Field(max_length=MAX_FAMILY_SIZE)
    right_family: list[PositiveInteger] = Field(max_length=MAX_FAMILY_SIZE)

    @model_validator(mode="after")
    def require_positive_families_and_graph_budget(self) -> Self:
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

    left_family: list[PositiveInteger]
    right_family: list[PositiveInteger]
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_FAMILY_SIZE",
    "MAX_GRAPH_EDGES",
    "MAX_TOTAL_FAMILY_SIZE",
    "DivisibilityIncidenceGraphRequest",
    "DivisibilityIncidenceGraphResult",
    "PositiveInteger",
]
