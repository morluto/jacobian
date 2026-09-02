"""Typed contracts for the free-tree enumeration operation."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

# A000055 gives 19,320 free trees at order 16. Order 17 has 48,629 retained
# graph rows and exceeds the owner-local family-cardinality envelope.
MAX_ORDER = 16
MAX_FREE_TREE_COUNT = 19_320


class FreeTreeEnumerationRequest(StrictModel):
    """Request to enumerate all non-isomorphic free trees of a given order."""

    order: int = Field(ge=0, le=MAX_ORDER)


class FreeTreeEnumerationResult(StrictModel):
    """A complete family of non-isomorphic free trees."""

    order: int = Field(ge=0, le=MAX_ORDER)
    trees: tuple[SimpleUndirectedGraph, ...] = Field(max_length=MAX_FREE_TREE_COUNT)
    count: int = Field(ge=0, le=MAX_FREE_TREE_COUNT)


__all__ = [
    "MAX_FREE_TREE_COUNT",
    "MAX_ORDER",
    "FreeTreeEnumerationRequest",
    "FreeTreeEnumerationResult",
]
