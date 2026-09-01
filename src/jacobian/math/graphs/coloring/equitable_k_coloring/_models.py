"""Typed contracts for the equitable k-colourability decision."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EQUITABLE_COLORING_SEARCH_NODES = 1_000_000


def _is_complete(graph: SimpleUndirectedGraph) -> bool:
    n = len(graph.vertices)
    return len(graph.edges) == n * (n - 1) // 2


class EquitableColoringRequest(StrictModel):
    """Request to decide equitable k-colourability."""

    graph: SimpleUndirectedGraph
    k: int = Field(gt=0)

    @model_validator(mode="after")
    def require_bounded_search(self) -> Self:
        n = len(self.graph.vertices)
        if (
            self.graph.edges
            and 0 < self.k < n
            and not _is_complete(self.graph)
            and self.k**n > MAX_EQUITABLE_COLORING_SEARCH_NODES
        ):
            raise PydanticCustomError(
                "graph.equitable_coloring_search_exceeded",
                "equitable coloring exceeds the 1000000-node search bound",
            )
        return self


class EquitableColoringResult(StrictModel):
    """The equitable k-colouring decision."""

    graph: SimpleUndirectedGraph
    k: int
    colorable: bool
    coloring: tuple[int, ...] | None = None


__all__ = [
    "MAX_EQUITABLE_COLORING_SEARCH_NODES",
    "EquitableColoringRequest",
    "EquitableColoringResult",
]
