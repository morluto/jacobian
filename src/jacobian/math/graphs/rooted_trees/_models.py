"""Private request models for rooted-tree operations."""

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_VERTICES,
    GraphVertexLabel,
    SimpleUndirectedGraph,
)


class RootedTreeFinePartitionRequest(StrictModel):
    """One graph, declared root, and requested maximum shrub order."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "The retained canonical finite simple undirected graph. The graph "
            "must be nonempty and have at most "
            f"{MAX_SIMPLE_GRAPH_VERTICES} vertices; every vertex label must be "
            "nonempty and use at most 64 UTF-8 bytes. A well-formed non-tree "
            "returns a typed NOT_A_TREE outcome."
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

    @model_validator(mode="after")
    def require_supported_order(self) -> Self:
        if len(self.graph.vertices) > MAX_SIMPLE_GRAPH_VERTICES:
            raise PydanticCustomError(
                "graph.rooted_tree.fine_partition.vertex_bound",
                "fine-partition construction supports at most "
                f"{MAX_SIMPLE_GRAPH_VERTICES} vertices",
            )
        return self


__all__ = ["RootedTreeFinePartitionRequest"]
