"""Typed contracts for the maximal-clique hypergraph operation."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_LABEL_LENGTH,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_MAXIMAL_CLIQUE_ENUMERATION_WORK = 100_000


def _moon_moser_bound(vertex_count: int) -> int:
    """Return the sharp maximum number of maximal cliques on ``vertex_count`` vertices."""
    if vertex_count <= 1:
        return 1
    quotient, remainder = divmod(vertex_count, 3)
    if remainder == 0:
        return int(3**quotient)
    if remainder == 1:
        return int(4 * 3 ** (quotient - 1))
    return int(2 * 3**quotient)


def _maximal_clique_admission_error(
    graph: SimpleUndirectedGraph,
) -> tuple[str, str] | None:
    if any(len(label) > MAX_LABEL_LENGTH for label in graph.vertices):
        return (
            "label_bound",
            f"graph vertex labels must be at most {MAX_LABEL_LENGTH} characters",
        )
    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    if edge_count == 0 or edge_count == vertex_count * (vertex_count - 1) // 2:
        return None
    clique_bound = _moon_moser_bound(vertex_count)
    if clique_bound > MAX_EDGES:
        return (
            "edge_bound",
            f"maximal-clique family may exceed the {MAX_EDGES}-edge result bound",
        )
    incidence_bound = vertex_count * clique_bound
    if incidence_bound > MAX_TOTAL_INCIDENCES:
        return (
            "incidence_bound",
            "maximal-clique family may exceed the "
            f"{MAX_TOTAL_INCIDENCES}-incidence result bound",
        )
    if incidence_bound > MAX_MAXIMAL_CLIQUE_ENUMERATION_WORK:
        return (
            "work_bound",
            "maximal-clique enumeration exceeds the admitted work bound",
        )
    return None


class MaximalCliqueHypergraphRequest(StrictModel):
    """Request to construct the maximal-clique hypergraph of a graph."""

    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_bounded_clique_family(self) -> Self:
        failure = _maximal_clique_admission_error(self.graph)
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"maximal_clique.{code}", message)
        return self


class MaximalCliqueHypergraphResult(StrictModel):
    """The maximal-clique hypergraph of the source graph."""

    graph: SimpleUndirectedGraph
    hypergraph: FiniteHypergraph
    clique_count: int


__all__ = [
    "MaximalCliqueHypergraphRequest",
    "MaximalCliqueHypergraphResult",
    "_maximal_clique_admission_error",
]
