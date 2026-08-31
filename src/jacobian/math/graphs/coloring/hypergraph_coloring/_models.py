"""Typed contracts for the hypergraph colouring decision operation."""

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_HYPERGRAPH_COLORING_SEARCH_STATES = 2_000_000


def _hypergraph_coloring_admission_error(
    hypergraph: FiniteHypergraph, palette_size: int
) -> tuple[str, str] | None:
    if palette_size < 1:
        return ("invalid_palette", "palette_size must be positive")
    vertex_count = len(hypergraph.vertices)
    if (
        not hypergraph.edges
        or any(len(members) <= 1 for _, members in hypergraph.edges)
        or palette_size <= 1
        or palette_size >= vertex_count
    ):
        return None
    incidence_work = max(
        (
            sum(len(members) for _, members in hypergraph.edges if vertex in members)
            for vertex in hypergraph.vertices
        ),
        default=1,
    )
    if (
        palette_size**vertex_count * incidence_work
        > MAX_HYPERGRAPH_COLORING_SEARCH_STATES
    ):
        return (
            "search_bound",
            "hypergraph coloring exceeds the admitted backtracking-state bound",
        )
    return None


class HypergraphColoringRequest(StrictModel):
    """Request to decide q-colourability of a hypergraph."""

    hypergraph: FiniteHypergraph
    palette_size: int

    @model_validator(mode="after")
    def require_bounded_search(self) -> Self:
        failure = _hypergraph_coloring_admission_error(
            self.hypergraph, self.palette_size
        )
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"hypergraph_coloring.{code}", message)
        return self


class HypergraphColoringResult(StrictModel):
    """The hypergraph colouring decision."""

    hypergraph: FiniteHypergraph
    palette_size: int
    colorable: bool
    coloring: tuple[int, ...] | None = None


__all__ = [
    "MAX_HYPERGRAPH_COLORING_SEARCH_STATES",
    "HypergraphColoringRequest",
    "HypergraphColoringResult",
    "_hypergraph_coloring_admission_error",
]
