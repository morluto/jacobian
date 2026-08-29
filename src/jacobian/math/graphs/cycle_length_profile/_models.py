"""Typed contracts for the cycle-length profile operation."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class CycleLengthProfileRequest(StrictModel):
    """Request the complete simple-cycle length profile of a graph."""

    graph: SimpleUndirectedGraph


class CycleLengthEntry(StrictModel):
    """One cycle length and a witness cycle."""

    length: int
    witness: tuple[str, ...]


class CycleLengthProfileResult(StrictModel):
    """The complete cycle-length profile of a graph."""

    graph: SimpleUndirectedGraph
    entries: tuple[CycleLengthEntry, ...]
    cycle_lengths: tuple[int, ...]


__all__ = [
    "CycleLengthEntry",
    "CycleLengthProfileRequest",
    "CycleLengthProfileResult",
]
