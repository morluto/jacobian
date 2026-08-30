"""Private request models for rooted-tree operations."""

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.graphs.values import GraphVertexLabel, SimpleUndirectedGraph


class RootedTreeFinePartitionRequest(StrictModel):
    """One graph, declared root, and requested maximum shrub order."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "The retained canonical finite simple undirected graph. The graph "
            "must be nonempty; every vertex label must be nonempty and use at "
            "most 64 UTF-8 bytes. A well-formed non-tree returns a typed "
            "NOT_A_TREE outcome."
        )
    )
    root: GraphVertexLabel = Field(
        description="A declared graph vertex used as the root of the tree."
    )
    component_size_limit: StrictInt = Field(
        ge=1,
        le=255,
        description=(
            "The inclusive maximum number of vertices in each returned shrub; "
            "it must be strictly smaller than the graph order."
        ),
    )


__all__ = ["RootedTreeFinePartitionRequest"]
