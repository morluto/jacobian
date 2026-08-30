"""Canonical source-bound values for rooted-tree fine partitions."""

from __future__ import annotations

from collections import deque
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


def _graph_topology(
    graph: SimpleUndirectedGraph,
    root: str,
) -> tuple[bool, bool, int, dict[str, int]]:
    adjacency = {vertex: set[str]() for vertex in graph.vertices}
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    component_count = 0
    depths: dict[str, int] = {}
    for start in (root, *graph.vertices):
        if start in depths:
            continue
        component_count += 1
        depths[start] = 0
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if neighbor not in depths:
                    depths[neighbor] = depths[vertex] + 1
                    queue.append(neighbor)

    connected = component_count == 1
    has_cycle = len(graph.edges) > len(graph.vertices) - component_count
    return connected, has_cycle, component_count, depths


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
    def require_source_representation(self) -> Self:  # noqa: C901
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

        connected, has_cycle, component_count, depths = _graph_topology(
            self.graph, self.root
        )
        outcome = self.outcome
        if isinstance(outcome, RootedTreeNotATree):
            if (
                outcome.connected != connected
                or outcome.has_cycle != has_cycle
                or outcome.component_count != component_count
            ):
                raise _fine_partition_error(
                    "non_tree_diagnostic",
                    "the NOT_A_TREE diagnostic must match the retained graph",
                )
            if connected and not has_cycle:
                raise _fine_partition_error(
                    "non_tree_status",
                    "a NOT_A_TREE outcome requires a disconnected or cyclic graph",
                )
            return self

        if not connected or has_cycle:
            raise _fine_partition_error(
                "constructed_graph_topology",
                "a CONSTRUCTED outcome requires a connected acyclic graph",
            )

        seeds_x = set(outcome.seeds_x)
        seeds_y = set(outcome.seeds_y)
        seeds = seeds_x | seeds_y
        if not seeds <= graph_vertices:
            raise _fine_partition_error(
                "seed_source_membership",
                "every seed must be a retained graph vertex",
            )
        if self.root not in seeds_x:
            raise _fine_partition_error(
                "root_seed", "the retained root must be an X seed"
            )
        if any(depths[seed] % 2 for seed in seeds_x) or any(
            depths[seed] % 2 == 0 for seed in seeds_y
        ):
            raise _fine_partition_error(
                "seed_parity",
                "X and Y seeds must match their rooted graph depth parity",
            )
        if len(seeds_x) * self.component_size_limit > 12 * (
            len(graph_vertices) - 1
        ) or len(seeds_y) * self.component_size_limit > 12 * (len(graph_vertices) - 1):
            raise _fine_partition_error(
                "seed_bound",
                "each seed side must satisfy the fine-partition size bound",
            )

        reported_vertices: set[str] = set()
        reported_edges: set[_Edge] = set()

        def report_edge(edge: _Edge, *, field: str) -> None:
            if edge not in self.graph.edges:
                raise _fine_partition_error(
                    f"{field}_source_membership",
                    f"every {field.replace('_', ' ')} must belong to the graph",
                )
            if edge in reported_edges:
                raise _fine_partition_error(
                    "edge_partition",
                    "each source edge must occur in exactly one outcome row",
                )
            reported_edges.add(edge)

        for edge in outcome.seed_edges:
            report_edge(edge, field="seed_edge")
            if not set(edge) <= seeds:
                raise _fine_partition_error(
                    "seed_edge_membership",
                    "seed edges must have both endpoints in the seed set",
                )

        for shrub in outcome.shrubs:
            vertices = set(shrub.vertices)
            if not vertices <= graph_vertices:
                raise _fine_partition_error(
                    "shrub_source_membership",
                    "every shrub vertex must be a retained graph vertex",
                )
            if vertices & seeds:
                raise _fine_partition_error(
                    "shrub_seed_disjoint",
                    "shrub vertices must be disjoint from all seeds",
                )
            if reported_vertices & vertices:
                raise _fine_partition_error(
                    "shrub_vertex_partition",
                    "shrub vertices must be pairwise disjoint",
                )
            reported_vertices.update(vertices)
            if len(vertices) > self.component_size_limit:
                raise _fine_partition_error(
                    "shrub_size",
                    "a shrub must not exceed component_size_limit",
                )
            shrub_adjacency = {vertex: set[str]() for vertex in vertices}
            for left, right in shrub.edges:
                if left in vertices and right in vertices:
                    shrub_adjacency[left].add(right)
                    shrub_adjacency[right].add(left)
            reached = {shrub.root_vertex}
            queue = deque([shrub.root_vertex])
            while queue:
                vertex = queue.popleft()
                for neighbor in shrub_adjacency[vertex]:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        queue.append(neighbor)
            if reached != vertices:
                raise _fine_partition_error(
                    "shrub_connected",
                    "each shrub must be one connected source component",
                )
            if shrub.root_vertex != min(
                vertices, key=lambda vertex: (depths[vertex], vertex)
            ):
                raise _fine_partition_error(
                    "shrub_root_vertex",
                    "a shrub root_vertex must be its rootward vertex",
                )
            if shrub.upper_seed not in seeds:
                raise _fine_partition_error(
                    "shrub_upper_seed_membership",
                    "a shrub upper_seed must be a retained seed",
                )
            if depths[shrub.upper_seed] != depths[shrub.root_vertex] - 1:
                raise _fine_partition_error(
                    "shrub_upper_seed",
                    "a shrub upper_seed must be the parent of root_vertex",
                )

            for edge in shrub.edges:
                report_edge(edge, field="shrub_edge")
                if not set(edge) <= vertices:
                    raise _fine_partition_error(
                        "shrub_edge_membership",
                        "shrub edges must have both endpoints in that shrub",
                    )

            boundary_seed_values: set[str] = set()
            for edge in shrub.boundary_edges:
                report_edge(edge, field="boundary_edge")
                endpoints = set(edge)
                if (
                    len(endpoints & vertices) != 1
                    or len(endpoints - vertices) != 1
                    or not (endpoints - vertices) <= seeds
                ):
                    raise _fine_partition_error(
                        "boundary_edge_membership",
                        "boundary edges must join one shrub vertex to one seed",
                    )
                boundary_seed_values.update(endpoints - vertices)
            if set(shrub.boundary_seeds) != boundary_seed_values:
                raise _fine_partition_error(
                    "boundary_seed_coverage",
                    "boundary_seeds must be exactly the seed endpoints of boundary_edges",
                )
            expected_route_side = (
                "X" if all(seed in seeds_x for seed in boundary_seed_values) else "Y"
            )
            if (
                not boundary_seed_values
                or (
                    not boundary_seed_values <= seeds_x
                    and not boundary_seed_values <= seeds_y
                )
                or shrub.route_side != expected_route_side
            ):
                raise _fine_partition_error(
                    "boundary_seed_side",
                    "a shrub boundary must be nonempty and lie in one seed side",
                )
            if shrub.upper_seed not in boundary_seed_values:
                raise _fine_partition_error(
                    "upper_seed_boundary",
                    "a shrub upper_seed must be incident to its boundary",
                )
            root_boundary_edge = tuple(sorted((shrub.upper_seed, shrub.root_vertex)))
            if root_boundary_edge not in shrub.boundary_edges:
                raise _fine_partition_error(
                    "upper_seed_parent",
                    "a shrub upper_seed must be the parent of root_vertex",
                )

        shrub_order = tuple(
            (depths[shrub.root_vertex], shrub.vertices) for shrub in outcome.shrubs
        )
        if shrub_order != tuple(sorted(shrub_order)):
            raise _fine_partition_error(
                "shrub_order",
                "shrubs must be ordered by root distance and vertex tuple",
            )

        if reported_vertices != graph_vertices - seeds:
            raise _fine_partition_error(
                "vertex_partition",
                "seeds and shrubs must partition all graph vertices",
            )
        if reported_edges != set(self.graph.edges):
            raise _fine_partition_error(
                "edge_partition",
                "seed, shrub, and boundary rows must partition all graph edges",
            )
        return self


__all__ = [
    "RootedTreeFinePartition",
    "RootedTreeFinePartitionConstructed",
    "RootedTreeNotATree",
    "RootedTreeShrub",
]
