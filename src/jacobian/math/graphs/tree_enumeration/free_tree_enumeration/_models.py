"""Typed contracts for the free-tree enumeration operation."""

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_ORDER = 20


class FreeTreeEnumerationRequest(StrictModel):
    """Request to enumerate all non-isomorphic free trees of a given order."""

    order: int


class FreeTreeEnumerationResult(StrictModel):
    """A complete family of non-isomorphic free trees."""

    order: int
    trees: tuple[SimpleUndirectedGraph, ...]
    count: int


__all__ = [
    "MAX_ORDER",
    "FreeTreeEnumerationRequest",
    "FreeTreeEnumerationResult",
]
