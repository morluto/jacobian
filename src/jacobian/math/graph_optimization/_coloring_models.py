"""Contracts for bounded exact graph-coloring exploration."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian._models import StrictModel

GraphVertex = Annotated[
    str,
    StringConstraints(min_length=1, max_length=256, strict=True),
]


class ChromaticGraph(StrictModel):
    """A bounded simple undirected graph, accepting either edge orientation."""

    graph_schema_version: Literal["1"] = "1"
    vertices: tuple[GraphVertex, ...] = Field(max_length=32)
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(max_length=496)

    @model_validator(mode="after")
    def require_simple_graph(self) -> Self:
        vertex_set = set(self.vertices)
        if len(vertex_set) != len(self.vertices):
            raise ValueError("graph vertices must be unique")
        normalized_edges = {tuple(sorted((left, right))) for left, right in self.edges}
        if any(left == right for left, right in self.edges):
            raise ValueError("graph edges must not contain self-loops")
        if any(
            left not in vertex_set or right not in vertex_set
            for left, right in self.edges
        ):
            raise ValueError("graph edges must reference declared vertices")
        if len(normalized_edges) != len(self.edges):
            raise ValueError("graph edges must be unique ignoring orientation")
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
            raise ValueError("result vertices must be unique")
        if self.order != len(self.vertices):
            raise ValueError("result order must match the vertex list")
        if self.lower_bound > self.upper_bound:
            raise ValueError("chromatic bounds must be ordered")
        if self.coloring is not None and set(self.coloring) != set(self.vertices):
            raise ValueError("coloring must assign every result vertex")
        if self.coloring is not None and any(
            color < 0 or color >= self.upper_bound for color in self.coloring.values()
        ):
            raise ValueError("coloring values must lie below the upper bound")
        if self.status == "EXACT":
            if (
                self.chromatic_number is None
                or self.lower_bound != self.chromatic_number
                or self.upper_bound != self.chromatic_number
                or self.coloring is None
                or self.solver_status not in {"SATISFIABLE", "SPECIAL_CASE"}
            ):
                raise ValueError("exact result evidence is incomplete")
        elif self.chromatic_number is not None:
            raise ValueError("unknown result cannot carry a chromatic number")
        return self
