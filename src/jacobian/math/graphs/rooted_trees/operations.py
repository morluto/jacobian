"""Constructive bounded operations for rooted trees."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs._networkx import graph_from_value
from jacobian.math.graphs.rooted_trees.values import (
    RootedTreeFinePartition,
    RootedTreeFinePartitionConstructed,
    RootedTreeNotATree,
    RootedTreeShrub,
)
from jacobian.math.graphs.values import MAX_GRAPH_LABEL_BYTES, SimpleUndirectedGraph

type _Adjacency = dict[str, tuple[str, ...]]

@dataclass(frozen=True, slots=True)
class _FinePartitionPlan:
    connected: bool
    has_cycle: bool
    component_count: int


def _plan_request(
    graph: SimpleUndirectedGraph, root: str, component_size_limit: int
) -> _FinePartitionPlan:
    if not graph.vertices:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.rooted_tree.fine_partition.nonempty_graph",
            message="fine-partition construction requires a nonempty graph",
        )
    if root not in graph.vertices:
        raise OperationDomainValidationError(
            location=("root",),
            code="graph.rooted_tree.fine_partition.root_membership",
            message="root must be a declared graph vertex",
        )
    if component_size_limit < 1 or component_size_limit >= len(graph.vertices):
        raise OperationDomainValidationError(
            location=("component_size_limit",),
            code="graph.rooted_tree.fine_partition.component_size_limit",
            message=(
                "component_size_limit must be at least 1 and strictly smaller "
                "than the graph order"
            ),
        )
    if any(not vertex for vertex in graph.vertices):
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.rooted_tree.fine_partition.empty_label",
            message="graph vertex labels must not be empty",
        )
    for vertex in graph.vertices:
        try:
            label_bytes = len(vertex.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise OperationDomainValidationError(
                location=("graph", "vertices"),
                code="graph.rooted_tree.fine_partition.label_utf8",
                message="every graph vertex label must be valid UTF-8 text",
            ) from exc
        if label_bytes > MAX_GRAPH_LABEL_BYTES:
            raise OperationDomainValidationError(
                location=("graph", "vertices"),
                code="graph.rooted_tree.fine_partition.label_bytes",
                message=(
                    f"every graph vertex label must use at most "
                    f"{MAX_GRAPH_LABEL_BYTES} UTF-8 bytes"
                ),
            )

    # The canonical graph owner bounds n by 256 and m by n-choose-2. A tree
    # result has m=n-1, at most n-1 shrubs, exactly n seed-or-shrub vertex rows,
    # exactly m classified edge rows, and at most m boundary-seed incidences.
    # Graph scans and materialized intermediates are O(n+m); lexical canonical
    # ordering is the only superlinear work.
    backend = graph_from_value(graph)
    components = tuple(nx.connected_components(backend))
    component_count = len(components)
    connected = component_count == 1
    has_cycle = len(graph.edges) > len(graph.vertices) - component_count

    return _FinePartitionPlan(
        connected=connected,
        has_cycle=has_cycle,
        component_count=component_count,
    )


def _tree_orientation(
    adjacency: _Adjacency, root: str
) -> tuple[
    dict[str, str | None], dict[str, int], dict[str, tuple[str, ...]], tuple[str, ...]
]:
    parent: dict[str, str | None] = {root: None}
    depth = {root: 0}
    children: dict[str, list[str]] = {vertex: [] for vertex in adjacency}
    order: list[str] = []
    stack = [root]
    while stack:
        vertex = stack.pop()
        order.append(vertex)
        for neighbor in reversed(adjacency[vertex]):
            if neighbor in parent:
                continue
            parent[neighbor] = vertex
            depth[neighbor] = depth[vertex] + 1
            children[vertex].append(neighbor)
            stack.append(neighbor)
    return (
        parent,
        depth,
        {vertex: tuple(sorted(row)) for vertex, row in children.items()},
        tuple(order),
    )


def _components_after_deleting(
    adjacency: _Adjacency, deleted: set[str]
) -> tuple[frozenset[str], ...]:
    seen = set(deleted)
    components: list[frozenset[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        seen.add(start)
        component = {start}
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in reversed(adjacency[vertex]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(frozenset(component))
    return tuple(components)


def _initial_seeds(
    *,
    root: str,
    component_size_limit: int,
    children: dict[str, tuple[str, ...]],
    order: tuple[str, ...],
) -> set[str]:
    """Build W1 by the repeated deepest-heavy-component rule."""

    seeds: set[str] = set()
    residual_size: dict[str, int] = {}
    for vertex in reversed(order):
        size = 1 + sum(residual_size[child] for child in children[vertex])
        if vertex != root and size > component_size_limit:
            # Resetting at a cut makes these charged residual blocks disjoint;
            # every non-root W1 seed therefore consumes at least k+1 vertices.
            seeds.add(vertex)
            residual_size[vertex] = 0
        else:
            residual_size[vertex] = size
    seeds.add(root)
    return seeds


def _branch_vertices(
    component: frozenset[str], attachments: set[str], adjacency: _Adjacency
) -> set[str]:
    """Return branch vertices of the minimal subtree spanning attachments."""

    if len(attachments) < 3:
        return set()
    active = set(component)
    degree = {
        vertex: sum(neighbor in active for neighbor in adjacency[vertex])
        for vertex in active
    }
    leaves = deque(
        sorted(
            vertex
            for vertex in active
            if vertex not in attachments and degree[vertex] <= 1
        )
    )
    while leaves:
        vertex = leaves.popleft()
        if vertex not in active or vertex in attachments or degree[vertex] > 1:
            continue
        active.remove(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in active:
                continue
            degree[neighbor] -= 1
            if neighbor not in attachments and degree[neighbor] <= 1:
                leaves.append(neighbor)
    return {
        vertex
        for vertex in active
        if sum(neighbor in active for neighbor in adjacency[vertex]) >= 3
    }


def _add_branch_seeds(adjacency: _Adjacency, initial_seeds: set[str]) -> set[str]:
    """Build W2 from branch vertices of each attachment-spanning subtree."""

    seeds = set(initial_seeds)
    for component in _components_after_deleting(adjacency, initial_seeds):
        attachments = {
            vertex
            for vertex in component
            if any(neighbor in initial_seeds for neighbor in adjacency[vertex])
        }
        # Leaf-pruning computes the unique attachment-spanning subtree. Its
        # degree-at-least-three vertices are bounded by attachments minus two,
        # which is the W2 charge used in the theorem.
        seeds.update(_branch_vertices(component, attachments, adjacency))
    return seeds


def _add_parity_seeds(
    *,
    adjacency: _Adjacency,
    branch_seeds: set[str],
    parent: dict[str, str | None],
    depth: dict[str, int],
) -> set[str]:
    """Build W3 using the two parity cuts from the fine-partition proof.

    This is the constructive W3 step of Hladký--Piguet, Lemma 5.3: cut the
    rootward vertex of every internal component whose upper seed is odd, and
    cut the parent of every odd lower seed. Both new cut types are even.
    """

    seeds = set(branch_seeds)
    lower_seeds_by_parent: dict[str, list[str]] = {}
    for seed in branch_seeds:
        seed_parent = parent[seed]
        if seed_parent is not None:
            lower_seeds_by_parent.setdefault(seed_parent, []).append(seed)
    for component in _components_after_deleting(adjacency, branch_seeds):
        lower_seeds = tuple(
            seed
            for vertex in component
            for seed in lower_seeds_by_parent.get(vertex, ())
        )
        if not lower_seeds:
            continue

        top = min(component, key=lambda vertex: (depth[vertex], vertex))
        upper_seed = parent[top]
        # Internal components inject into lower W2 seeds, and each odd lower
        # seed has one parent. Thus these two additions contribute at most two
        # new vertices per W2 seed, yielding |W3| <= 3|W2|.
        if upper_seed is not None and depth[upper_seed] % 2 == 1:
            seeds.add(top)
        for lower_seed in lower_seeds:
            lower_parent = parent[lower_seed]
            if depth[lower_seed] % 2 == 1 and lower_parent is not None:
                seeds.add(lower_parent)
    return seeds


def _constructed_outcome(
    *,
    graph: SimpleUndirectedGraph,
    adjacency: _Adjacency,
    seeds: set[str],
    parent: dict[str, str | None],
    depth: dict[str, int],
) -> RootedTreeFinePartitionConstructed:
    seeds_x = tuple(sorted(seed for seed in seeds if depth[seed] % 2 == 0))
    seeds_y = tuple(sorted(seeds - set(seeds_x)))
    components = tuple(
        sorted(
            _components_after_deleting(adjacency, seeds),
            key=lambda component: (
                min(depth[vertex] for vertex in component),
                tuple(sorted(component)),
            ),
        )
    )
    owner = {
        vertex: index
        for index, component in enumerate(components)
        for vertex in component
    }
    seed_edges: list[tuple[str, str]] = []
    shrub_edges: list[list[tuple[str, str]]] = [[] for _ in components]
    boundary_edges: list[list[tuple[str, str]]] = [[] for _ in components]
    for edge in graph.edges:
        left, right = edge
        left_seed = left in seeds
        right_seed = right in seeds
        if left_seed and right_seed:
            seed_edges.append(edge)
        elif left_seed:
            boundary_edges[owner[right]].append(edge)
        elif right_seed:
            boundary_edges[owner[left]].append(edge)
        else:
            shrub_edges[owner[left]].append(edge)

    shrubs: list[RootedTreeShrub] = []
    for component_index, component in enumerate(components):
        vertices = tuple(sorted(component))
        edges = tuple(sorted(shrub_edges[component_index]))
        component_boundary_edges = tuple(sorted(boundary_edges[component_index]))
        boundary_seeds = tuple(
            sorted(
                {
                    right if left in component else left
                    for left, right in component_boundary_edges
                }
            )
        )
        top = min(component, key=lambda vertex: (depth[vertex], vertex))
        upper_seed = parent[top]
        if upper_seed is None:
            raise RuntimeError("a non-seed shrub must have an upper seed")
        route_side = "X" if depth[boundary_seeds[0]] % 2 == 0 else "Y"
        if any(
            depth[seed] % 2 != depth[boundary_seeds[0]] % 2 for seed in boundary_seeds
        ):
            raise RuntimeError(
                "fine-partition parity construction produced a mixed boundary"
            )
        shrubs.append(
            RootedTreeShrub.model_construct(
                index=component_index,
                vertices=vertices,
                edges=edges,
                boundary_seeds=boundary_seeds,
                boundary_edges=component_boundary_edges,
                upper_seed=upper_seed,
                root_vertex=top,
                route_side=route_side,
            )
        )
    return RootedTreeFinePartitionConstructed.model_construct(
        status="CONSTRUCTED",
        seeds_x=seeds_x,
        seeds_y=seeds_y,
        seed_edges=tuple(sorted(seed_edges)),
        shrubs=tuple(shrubs),
    )


def construct_fine_partition(
    graph: SimpleUndirectedGraph,
    root: str,
    component_size_limit: int,
) -> RootedTreeFinePartition:
    """Construct a deterministic bounded fine partition of a rooted tree.

    A well-formed graph that is disconnected or contains a cycle returns a
    typed ``NOT_A_TREE`` diagnostic. For a tree, the result reconstructs the
    complete source from seed, shrub, and boundary edge rows; each shrub has at
    most ``component_size_limit`` vertices, each shrub boundary belongs to one
    seed parity class, and each seed class has size at most
    ``12 * (n - 1) / component_size_limit``.

    The constructed partition is deterministic for the supplied labels and
    root, but is not claimed to be invariant under relabelling. Public vertex
    and edge axes are lexically sorted; shrubs are indexed by increasing root
    distance of ``root_vertex``, with their vertex tuple as the tie-breaker.
    """

    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("graph must be a SimpleUndirectedGraph")
    if not isinstance(root, str):
        raise TypeError("root must be a string")
    if isinstance(component_size_limit, bool) or not isinstance(
        component_size_limit, int
    ):
        raise TypeError("component_size_limit must be an integer")
    plan = _plan_request(graph, root, component_size_limit)

    if not plan.connected or plan.has_cycle:
        non_tree_outcome = RootedTreeNotATree.model_construct(
            status="NOT_A_TREE",
            connected=plan.connected,
            has_cycle=plan.has_cycle,
            component_count=plan.component_count,
        )
        return RootedTreeFinePartition._from_kernel(
            graph=graph,
            root=root,
            component_size_limit=component_size_limit,
            outcome=non_tree_outcome,
        )

    mutable_adjacency: dict[str, list[str]] = {vertex: [] for vertex in graph.vertices}
    for left, right in graph.edges:
        mutable_adjacency[left].append(right)
        mutable_adjacency[right].append(left)
    adjacency = {
        vertex: tuple(sorted(neighbors))
        for vertex, neighbors in mutable_adjacency.items()
    }
    parent, depth, children, order = _tree_orientation(adjacency, root)
    initial_seeds = _initial_seeds(
        root=root,
        component_size_limit=component_size_limit,
        children=children,
        order=order,
    )
    branch_seeds = _add_branch_seeds(adjacency, initial_seeds)
    seeds = _add_parity_seeds(
        adjacency=adjacency,
        branch_seeds=branch_seeds,
        parent=parent,
        depth=depth,
    )
    constructed_outcome = _constructed_outcome(
        graph=graph,
        adjacency=adjacency,
        seeds=seeds,
        parent=parent,
        depth=depth,
    )
    return RootedTreeFinePartition._from_kernel(
        graph=graph,
        root=root,
        component_size_limit=component_size_limit,
        outcome=constructed_outcome,
    )


__all__ = ["construct_fine_partition"]
