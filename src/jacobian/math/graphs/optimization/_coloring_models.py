"""Contracts for bounded exact graph-coloring exploration."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._graph_validation import (
    require_simple_undirected_graph,
)

GraphVertex = Annotated[
    str,
    StringConstraints(min_length=1, max_length=256, strict=True),
]


class ChromaticGraph(StrictModel):
    """A bounded simple undirected graph, accepting either edge orientation."""

    vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=496)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        require_simple_undirected_graph(self.vertices, self.edges)
        return self


class PolynomialTimeGraph(StrictModel):
    """A bounded simple undirected graph for polynomial-time invariants.

    Mirrors :class:`ChromaticGraph` validation but with a larger envelope
    (256 vertices, 32640 edges = 256*255/2) suitable for NetworkX
    polynomial-time operations such as girth, edge/vertex connectivity,
    Eulerian check, spanning-tree count, maximum matching, k-core, and
    distance matrix. NP-hard Z3 operations continue to use
    :class:`ChromaticGraph` (32 vertices) or their own tighter budgets.
    """

    vertices: tuple[GraphVertex, ...] = Field(max_length=256)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=32640)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        require_simple_undirected_graph(self.vertices, self.edges)
        return self


class ChromaticNumberBudget(StrictModel):
    """Total wall-clock budget for the bounded coloring search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)


class GraphChromaticNumberRequest(StrictModel):
    """Request one bounded exact chromatic-number exploration."""

    graph: ChromaticGraph
    resource_budget: ChromaticNumberBudget = Field(
        default_factory=ChromaticNumberBudget
    )


class ChromaticSearchStep(StrictModel):
    """One k-colorability decision made by the solver."""

    colors: StrictInt = Field(ge=1, le=32)
    status: Literal["SATISFIABLE", "UNSATISFIABLE", "UNKNOWN"]


class GraphChromaticNumberOutput(StrictModel):
    """Exact result or bounded non-conclusion with inspectable evidence."""

    status: Literal["EXACT", "UNKNOWN"]
    vertices: tuple[GraphVertex, ...]
    order: StrictInt = Field(ge=0, le=32)
    chromatic_number: StrictInt | None = Field(default=None, ge=0, le=32)
    lower_bound: StrictInt = Field(ge=0, le=32)
    upper_bound: StrictInt = Field(ge=0, le=32)
    coloring: dict[GraphVertex, StrictInt] | None = None
    solver_status: Literal["SATISFIABLE", "UNKNOWN", "SPECIAL_CASE"]
    tested: tuple[ChromaticSearchStep, ...]
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_result_status(self) -> Self:
        if len(set(self.vertices)) != len(self.vertices):
            raise PydanticCustomError(
                "graph.result_vertices_must_be_unique", "result vertices must be unique"
            )
        if self.order != len(self.vertices):
            raise PydanticCustomError(
                "graph.result_order_must_match_the_vertex_list",
                "result order must match the vertex list",
            )
        if self.lower_bound > self.upper_bound:
            raise PydanticCustomError(
                "graph.chromatic_bounds_must_be_ordered",
                "chromatic bounds must be ordered",
            )
        if self.coloring is not None and set(self.coloring) != set(self.vertices):
            raise PydanticCustomError(
                "graph.coloring_must_assign_every_result_vertex",
                "coloring must assign every result vertex",
            )
        if self.coloring is not None and any(
            color < 0 or color >= self.upper_bound for color in self.coloring.values()
        ):
            raise PydanticCustomError(
                "graph.coloring_values_must_lie_below_the_upper_bound",
                "coloring values must lie below the upper bound",
            )
        if self.status == "EXACT":
            if (
                self.chromatic_number is None
                or self.lower_bound != self.chromatic_number
                or self.upper_bound != self.chromatic_number
                or self.coloring is None
                or self.solver_status not in {"SATISFIABLE", "SPECIAL_CASE"}
            ):
                raise PydanticCustomError(
                    "graph.exact_result_evidence_is_incomplete",
                    "exact result evidence is incomplete",
                )
        elif self.chromatic_number is not None:
            raise PydanticCustomError(
                "graph.unknown_result_cannot_carry_a_chromatic_number",
                "unknown result cannot carry a chromatic number",
            )
        return self
