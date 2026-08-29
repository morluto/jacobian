"""Canonical source-bound values for rooted-tree fine partitions."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_GRAPH_LABEL_BYTES,
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    GraphVertexLabel,
    SimpleUndirectedGraph,
)

_MAX_TREE_EDGES = MAX_INDEXED_SIMPLE_GRAPH_VERTICES - 1
type _Edge = tuple[str, str]


def _fine_partition_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"graph.rooted_tree.fine_partition.{code}", message)


def _require_sorted_unique(values: tuple[str, ...], *, field: str) -> None:
    if values != tuple(sorted(set(values))):
        raise _fine_partition_error(
            f"canonical_{field.replace(' ', '_')}",
            f"{field} must be unique and lexically sorted",
        )


def _require_sorted_unique_edges(values: tuple[_Edge, ...], *, field: str) -> None:
    if values != tuple(sorted(set(values))):
        raise _fine_partition_error(
            f"canonical_{field.replace(' ', '_')}",
            f"{field} must be unique and lexically sorted",
        )


class RootedTreeShrub(StrictModel):
    """One rooted component left after deleting all fine-partition seeds.

    ``root_vertex`` is the unique shrub vertex closest to the retained root,
    and ``upper_seed`` is its parent. ``route_side`` is the common bipartition
    side of every boundary seed.
    """

    index: StrictInt = Field(ge=0, le=MAX_INDEXED_SIMPLE_GRAPH_VERTICES - 1)
    vertices: tuple[GraphVertexLabel, ...] = Field(
        min_length=1, max_length=MAX_INDEXED_SIMPLE_GRAPH_VERTICES - 1
    )
    edges: tuple[_Edge, ...] = Field(max_length=_MAX_TREE_EDGES)
    boundary_seeds: tuple[GraphVertexLabel, ...] = Field(
        min_length=1, max_length=MAX_INDEXED_SIMPLE_GRAPH_VERTICES
    )
    boundary_edges: tuple[_Edge, ...] = Field(min_length=1, max_length=_MAX_TREE_EDGES)
    upper_seed: GraphVertexLabel
    root_vertex: GraphVertexLabel
    route_side: Literal["X", "Y"]

    @model_validator(mode="after")
    def require_canonical_rows(self) -> Self:
        _require_sorted_unique(self.vertices, field="shrub vertices")
        _require_sorted_unique_edges(self.edges, field="shrub edges")
        _require_sorted_unique(self.boundary_seeds, field="boundary seeds")
        _require_sorted_unique_edges(self.boundary_edges, field="boundary edges")
        if self.root_vertex not in self.vertices:
            raise _fine_partition_error(
                "root_vertex_membership",
                "a shrub root_vertex must belong to that shrub",
            )
        if self.upper_seed not in self.boundary_seeds:
            raise _fine_partition_error(
                "upper_seed_membership",
                "a shrub upper_seed must belong to its boundary",
            )
        return self


class RootedTreeFinePartitionConstructed(StrictModel):
    """The constructed seed sides and all source-bound shrubs."""

    status: Literal["CONSTRUCTED"] = "CONSTRUCTED"
    seeds_x: tuple[GraphVertexLabel, ...] = Field(
        max_length=MAX_INDEXED_SIMPLE_GRAPH_VERTICES
    )
    seeds_y: tuple[GraphVertexLabel, ...] = Field(
        max_length=MAX_INDEXED_SIMPLE_GRAPH_VERTICES
    )
    seed_edges: tuple[_Edge, ...] = Field(max_length=_MAX_TREE_EDGES)
    shrubs: tuple[RootedTreeShrub, ...] = Field(
        max_length=MAX_INDEXED_SIMPLE_GRAPH_VERTICES - 1
    )

    @model_validator(mode="after")
    def require_canonical_rows(self) -> Self:
        _require_sorted_unique(self.seeds_x, field="X seeds")
        _require_sorted_unique(self.seeds_y, field="Y seeds")
        _require_sorted_unique_edges(self.seed_edges, field="seed edges")
        if set(self.seeds_x) & set(self.seeds_y):
            raise _fine_partition_error(
                "disjoint_seed_sides", "the X and Y seed sides must be disjoint"
            )
        if tuple(shrub.index for shrub in self.shrubs) != tuple(
            range(len(self.shrubs))
        ):
            raise _fine_partition_error(
                "shrub_indexes", "shrub indexes must be consecutive from zero"
            )
        return self


class RootedTreeNotATree(StrictModel):
    """A total diagnostic for a well-formed graph that is not a tree."""

    status: Literal["NOT_A_TREE"] = "NOT_A_TREE"
    connected: StrictBool
    has_cycle: StrictBool
    component_count: StrictInt = Field(ge=1, le=MAX_INDEXED_SIMPLE_GRAPH_VERTICES)


RootedTreeFinePartitionOutcome = Annotated[
    RootedTreeFinePartitionConstructed | RootedTreeNotATree,
    Field(discriminator="status"),
]


class RootedTreeFinePartition(StrictModel):
    """A source-bound fine partition or an exact non-tree diagnostic.

    In a constructed result the seeds contain the declared root, every shrub
    is a nonempty component of the source tree after seed deletion, and every
    source edge occurs exactly once as a seed edge, shrub edge, or boundary
    edge. Each shrub has at most ``component_size_limit`` vertices. Its
    boundary is nonempty and lies wholly in the reported route side. The X and
    Y seed sides are the even- and odd-depth seed classes, respectively, and
    each contains at most ``12 * (n - 1) / component_size_limit`` vertices.
    Seed and edge axes are lexically sorted. Shrubs are indexed by increasing
    root distance of ``root_vertex``, then by their complete vertex tuple. This
    is deterministic for the retained labels, not label-independent canonical
    graph structure.
    """

    graph: SimpleUndirectedGraph
    root: GraphVertexLabel
    component_size_limit: StrictInt = Field(
        ge=1, le=MAX_INDEXED_SIMPLE_GRAPH_VERTICES - 1
    )
    outcome: RootedTreeFinePartitionOutcome

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        root: str,
        component_size_limit: int,
        outcome: RootedTreeFinePartitionConstructed | RootedTreeNotATree,
    ) -> Self:
        """Build an owner-produced value without replaying source binding."""

        return cls.model_construct(
            graph=graph,
            root=root,
            component_size_limit=component_size_limit,
            outcome=outcome,
        )

    @model_validator(mode="after")
    def require_source_representation(self) -> Self:
        graph_vertices = set(self.graph.vertices)
        if self.root not in graph_vertices:
            raise _fine_partition_error(
                "root_membership", "root must be a declared graph vertex"
            )
        if self.component_size_limit >= len(graph_vertices):
            raise _fine_partition_error(
                "component_size_limit",
                "component_size_limit must be strictly smaller than graph order",
            )
        if any(not vertex for vertex in self.graph.vertices):
            raise _fine_partition_error(
                "empty_label", "graph vertex labels must not be empty"
            )
        for vertex in self.graph.vertices:
            try:
                label_bytes = len(vertex.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise _fine_partition_error(
                    "label_utf8", "graph vertex labels must be valid UTF-8 text"
                ) from exc
            if label_bytes > MAX_GRAPH_LABEL_BYTES:
                raise _fine_partition_error(
                    "label_bytes",
                    "graph vertex labels must use at most "
                    f"{MAX_GRAPH_LABEL_BYTES} UTF-8 bytes",
                )
        return self


__all__ = [
    "RootedTreeFinePartition",
    "RootedTreeFinePartitionConstructed",
    "RootedTreeNotATree",
    "RootedTreeShrub",
]
