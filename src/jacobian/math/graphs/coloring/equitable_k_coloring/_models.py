"""Typed contracts for the equitable k-colourability decision."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class EquitableColoringRequest(StrictModel):
    """Request to decide equitable k-colourability."""

    graph: SimpleUndirectedGraph
    k: int


class EquitableColoringResult(StrictModel):
    """The equitable k-colouring decision."""

    graph: SimpleUndirectedGraph
    k: int
    colorable: bool
    coloring: tuple[int, ...] | None = None


__all__ = [
    "EquitableColoringRequest",
    "EquitableColoringResult",
]
