"""Typed contracts for edge-colouring Ramsey arrowing decision."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.coloring._models import EdgeColoringAssignment
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

# The graph value already bounds the carrier at 256 vertices.  Arrowing
# admission is otherwise controlled by the derived coloring/embedding work.
MAX_HOST_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES
MAX_HOST_EDGES = MAX_INDEXED_SIMPLE_GRAPH_EDGES
MAX_TARGET_VERTICES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES
MAX_TARGET_COUNT = 8
MAX_ARROWING_WORK = 20_000_000


def _validate_arrowing_envelope(
    host_graph: SimpleUndirectedGraph,
    targets: tuple[SimpleUndirectedGraph, ...],
) -> None:
    """Validate source structure and the complete finite-search envelope."""
    if not targets:
        raise PydanticCustomError(
            "graph.arrowing.no_targets", "at least one target graph is required"
        )
    if len(targets) > MAX_TARGET_COUNT:
        raise PydanticCustomError(
            "graph.arrowing.too_many_targets",
            f"at most {MAX_TARGET_COUNT} target graphs are supported",
        )
    if len(host_graph.vertices) > MAX_HOST_VERTICES:
        raise PydanticCustomError(
            "graph.arrowing.host_too_large",
            f"host graph has at most {MAX_HOST_VERTICES} vertices",
        )
    if len(host_graph.edges) > MAX_HOST_EDGES:
        raise PydanticCustomError(
            "graph.arrowing.host_too_many_edges",
            f"host graph has at most {MAX_HOST_EDGES} edges",
        )
    for i, target in enumerate(targets):
        if len(target.vertices) > MAX_TARGET_VERTICES:
            raise PydanticCustomError(
                "graph.arrowing.target_too_large",
                f"target {i} has at most {MAX_TARGET_VERTICES} vertices",
            )
        if not target.vertices:
            raise PydanticCustomError(
                "graph.arrowing.target_empty",
                f"target {i} must not be empty",
            )

    # Each colouring is checked against every target embedding.  The count is
    # deliberately conservative; permutations(n, t) bounds the embedding
    # checks for target t and includes sparse targets safely.
    host_vertices = len(host_graph.vertices)
    embedding_checks = sum(
        _permutation_upper_bound(host_vertices, len(target.vertices))
        for target in targets
    )
    work = len(targets) ** len(host_graph.edges) * embedding_checks
    if work > MAX_ARROWING_WORK:
        raise PydanticCustomError(
            "graph.arrowing.work_too_large",
            f"arrowing search requires at most {MAX_ARROWING_WORK} embedding checks",
        )


def _permutation_upper_bound(n: int, k: int) -> int:
    if k > n:
        return 0
    result = 1
    for value in range(n - k + 1, n + 1):
        result *= value
    return result


class EdgeColoringArrowingRequest(StrictModel):
    """Decide whether a host graph arrows a tuple of target graphs."""

    host_graph: SimpleUndirectedGraph
    targets: tuple[SimpleUndirectedGraph, ...] = Field(max_length=MAX_TARGET_COUNT)


class EdgeColoringArrowingResult(StrictModel):
    """Result of an edge-colouring Ramsey arrowing decision."""

    host_graph: SimpleUndirectedGraph
    targets: tuple[SimpleUndirectedGraph, ...]
    outcome: Literal["ARROWS", "DOES_NOT_ARROW"]
    avoiding_coloring: EdgeColoringAssignment | None = None

    @model_validator(mode="after")
    def require_outcome_certificate_shape(self) -> Self:
        if self.outcome == "ARROWS" and self.avoiding_coloring is not None:
            raise PydanticCustomError(
                "graph.arrowing.arrows_has_avoiding_coloring",
                "ARROWS results must not carry an avoiding colouring",
            )
        if self.outcome == "DOES_NOT_ARROW":
            if self.avoiding_coloring is None:
                raise PydanticCustomError(
                    "graph.arrowing.missing_avoiding_coloring",
                    "DOES_NOT_ARROW results must carry an avoiding colouring",
                )
            if self.avoiding_coloring.graph != self.host_graph:
                raise PydanticCustomError(
                    "graph.arrowing.mismatched_avoiding_coloring",
                    "avoiding colouring must be bound to the host graph",
                )
            if len(self.avoiding_coloring.coloring) != len(self.host_graph.edges):
                raise PydanticCustomError(
                    "graph.arrowing.incomplete_avoiding_coloring",
                    "avoiding colouring must cover every host edge",
                )
            if self.avoiding_coloring.colors != len(self.targets):
                raise PydanticCustomError(
                    "graph.arrowing.mismatched_avoiding_palette",
                    "avoiding colouring palette must match the target count",
                )
            if any(
                not 0 <= color < len(self.targets)
                for color in self.avoiding_coloring.coloring
            ):
                raise PydanticCustomError(
                    "graph.arrowing.invalid_avoiding_coloring",
                    "avoiding colouring must use ordered edge indices and valid colours",
                )
        return self


__all__ = [
    "MAX_ARROWING_WORK",
    "MAX_HOST_EDGES",
    "MAX_HOST_VERTICES",
    "MAX_TARGET_COUNT",
    "MAX_TARGET_VERTICES",
    "EdgeColoringArrowingRequest",
    "EdgeColoringArrowingResult",
    "_validate_arrowing_envelope",
]
