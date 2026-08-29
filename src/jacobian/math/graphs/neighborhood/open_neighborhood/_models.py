"""Typed contracts for the open neighbourhood operation."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class OpenNeighborhoodRequest(StrictModel):
    """Request for the open neighbourhood of a selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...]


class OpenNeighborhoodResult(StrictModel):
    """The complete open neighbourhood of the selected vertex set."""

    graph: SimpleUndirectedGraph
    selected_vertices: tuple[str, ...]
    neighborhood: tuple[str, ...]


__all__ = [
    "OpenNeighborhoodRequest",
    "OpenNeighborhoodResult",
]
