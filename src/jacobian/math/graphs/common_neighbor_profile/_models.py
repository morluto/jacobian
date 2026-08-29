"""Typed contracts for the common-neighbour profile operation."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class CommonNeighborProfileRequest(StrictModel):
    """Request the common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph


class PairEntry(StrictModel):
    """One unordered vertex pair with its common-neighbour set."""

    u: str
    v: str
    common_neighbors: tuple[str, ...]
    codegree: int


class CommonNeighborProfileResult(StrictModel):
    """The complete common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph
    pairs: tuple[PairEntry, ...]
    max_codegree: int
    histogram: tuple[int, ...]
    is_c4_free: bool


__all__ = [
    "CommonNeighborProfileRequest",
    "CommonNeighborProfileResult",
    "PairEntry",
]
