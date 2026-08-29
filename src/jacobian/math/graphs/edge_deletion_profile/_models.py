"""Typed contracts for the edge-deletion chromatic profile operation."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph


class EdgeDeletionProfileRequest(StrictModel):
    """Request the edge-deletion chromatic profile of a graph."""

    graph: SimpleUndirectedGraph
    deletion_order: int


class DeletionEntry(StrictModel):
    """One deleted edge set and the resulting chromatic number."""

    deleted_edges: tuple[tuple[str, str], ...]
    chromatic_number: int


class EdgeDeletionProfileResult(StrictModel):
    """The complete edge-deletion chromatic profile."""

    graph: SimpleUndirectedGraph
    source_chromatic_number: int
    deletion_order: int
    entries: tuple[DeletionEntry, ...]


__all__ = [
    "DeletionEntry",
    "EdgeDeletionProfileRequest",
    "EdgeDeletionProfileResult",
]
