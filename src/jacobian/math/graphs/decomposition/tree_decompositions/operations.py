"""Exact native kernels over tree decompositions.

All functions are deterministic and complete for accepted values.
"""

from __future__ import annotations

from collections import deque

from jacobian.math.graphs.values import SimpleUndirectedGraph

from ._models import (
    Adhesion,
    AdhesionsResult,
    BagIntersectionGraphResult,
    BagNode,
    OccurrenceSubtree,
    RerootResult,
    VertexOccurrencesResult,
    WidthResult,
)
from .values import TreeDecomposition

__all__ = [
    "adhesions",
    "bag_intersection_graph",
    "reroot",
    "restrict",
    "vertex_occurrences",
    "width",
]


def _index_of(td: TreeDecomposition) -> dict[str, int]:
    return {label: i for i, label in enumerate(td.tree_nodes)}


def _int_edges(td: TreeDecomposition) -> list[tuple[int, int]]:
    idx = _index_of(td)
    return [(idx[a], idx[b]) for a, b in td.tree_edges]


def width(td: TreeDecomposition) -> WidthResult:
    """Return bag cardinality per tree node, maximum bag cardinality, width
    (max bag cardinality minus 1), and the maximum-bag node labels."""
    bag_sizes = [len(bag) for bag in td.bags]
    max_size = max(bag_sizes)
    max_nodes = [td.tree_nodes[i] for i, s in enumerate(bag_sizes) if s == max_size]
    return WidthResult(
        bag_sizes=tuple(bag_sizes),
        max_bag_cardinality=max_size,
        width=max_size - 1,
        maximum_bag_nodes=tuple(max_nodes),
    )


def _occurrence_subtree(
    td: TreeDecomposition,
    adjacency: dict[int, list[int]],
    int_edges: list[tuple[int, int]],
    vertex: str,
    containing: list[int],
) -> OccurrenceSubtree:
    containing_set = set(containing)
    reached: set[int] = {containing[0]}
    stack = [containing[0]]
    while stack:
        current = stack.pop()
        for nxt in adjacency[current]:
            if nxt in containing_set and nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    nodes = sorted(reached)
    node_edges: list[tuple[str, str]] = []
    induced_degree: dict[int, int] = dict.fromkeys(reached, 0)
    for a, b in int_edges:
        if a in reached and b in reached:
            la, lb = td.tree_nodes[a], td.tree_nodes[b]
            node_edges.append((la, lb) if la <= lb else (lb, la))
            induced_degree[a] += 1
            induced_degree[b] += 1
    leaves = tuple(td.tree_nodes[n] for n in nodes if induced_degree[n] <= 1)
    return OccurrenceSubtree(
        nodes=tuple(td.tree_nodes[n] for n in nodes),
        edges=tuple(node_edges),
        count=len(nodes),
        leaves=leaves,
    )


def vertex_occurrences(
    td: TreeDecomposition,
) -> VertexOccurrencesResult:
    """Return per-source-vertex occurrence subtree node set, induced tree edges,
    occurrence counts, and leaf/extremal nodes."""
    int_edges = _int_edges(td)
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(td.tree_nodes))}
    for a, b in int_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    per_vertex: dict[str, OccurrenceSubtree] = {}
    for vertex in td.graph.vertices:
        containing = [i for i, bag in enumerate(td.bags) if vertex in bag]
        if not containing:
            per_vertex[vertex] = OccurrenceSubtree(
                nodes=(), edges=(), count=0, leaves=()
            )
            continue
        per_vertex[vertex] = _occurrence_subtree(
            td, adjacency, int_edges, vertex, containing
        )
    return VertexOccurrencesResult(per_vertex=per_vertex)


def adhesions(td: TreeDecomposition) -> AdhesionsResult:
    """For every decomposition-tree edge, compute adhesion(t,t') = B_t ∩ B_t',
    its size, and the left/right component vertex coverage after deleting tt'.

    Return the maximum adhesion, size profile, and exact separator sets. The
    result is a structural profile of the supplied decomposition, not a
    minimum-separator computation."""
    int_edges = _int_edges(td)
    bag_sets = [set(bag) for bag in td.bags]
    per_edge: list[Adhesion] = []
    for a, b in int_edges:
        la, lb = td.tree_nodes[a], td.tree_nodes[b]
        edge_label = (la, lb) if la <= lb else (lb, la)
        adhesion = sorted(bag_sets[a] & bag_sets[b])
        per_edge.append(
            Adhesion(
                edge=edge_label,
                adhesion=tuple(adhesion),
                size=len(adhesion),
            )
        )
    max_adhesion = max((row.size for row in per_edge), default=0)
    return AdhesionsResult(
        edges=tuple(per_edge),
        max_adhesion=max_adhesion,
        size_profile=tuple(row.size for row in per_edge),
    )


def _bfs_parents(
    adjacency: dict[int, list[int]], root_index: int
) -> tuple[dict[int, int | None], dict[int, int]]:
    parent: dict[int, int | None] = {root_index: None}
    depth: dict[int, int] = {root_index: 0}
    queue = deque([root_index])
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if nxt not in parent:
                parent[nxt] = current
                depth[nxt] = depth[current] + 1
                queue.append(nxt)
    return parent, depth


def _root_to_node_paths(
    adjacency: dict[int, list[int]],
    td: TreeDecomposition,
    parent: dict[int, int | None],
    root_index: int,
    root: str,
) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {root: [root]}
    queue = deque(adjacency[root_index])
    visited_paths: set[int] = {root_index}
    while queue:
        current = queue.popleft()
        if current in visited_paths:
            continue
        visited_paths.add(current)
        current_label = td.tree_nodes[current]
        parent_index = parent[current]
        if parent_index is None:
            raise RuntimeError("a non-root tree node has no parent")
        current_parent = td.tree_nodes[parent_index]
        paths[current_label] = paths[current_parent] + [current_label]
        for nxt in adjacency[current]:
            if nxt not in visited_paths:
                queue.append(nxt)
    return paths


def reroot(td: TreeDecomposition, root: str) -> RerootResult:
    """Return the same underlying decomposition rerooted at the selected tree
    node. Changing the root does not change the width, bags, or unrooted
    tree."""
    idx = _index_of(td)
    if root not in idx:
        raise ValueError("root must be a declared tree node")
    int_edges = _int_edges(td)
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(td.tree_nodes))}
    for a, b in int_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    root_index = idx[root]
    parent, depth = _bfs_parents(adjacency, root_index)
    parent_map: dict[str, str | None] = {}
    for node, parent_node in parent.items():
        parent_map[td.tree_nodes[node]] = (
            None if parent_node is None else td.tree_nodes[parent_node]
        )
    children_map: dict[str, tuple[str, ...]] = {
        td.tree_nodes[i]: () for i in range(len(td.tree_nodes))
    }
    for child, parent_node in parent.items():
        if parent_node is not None:
            children_map[td.tree_nodes[parent_node]] = (
                *children_map.get(td.tree_nodes[parent_node], ()),
                td.tree_nodes[child],
            )
    paths = _root_to_node_paths(adjacency, td, parent, root_index, root)
    return RerootResult(
        root=root,
        parent=parent_map,
        children=children_map,
        depth={td.tree_nodes[i]: depth[i] for i in depth},
        paths=paths,
    )


def _prune_redundant_leaves(
    active_nodes: set[int],
    adjacency: dict[int, list[int]],
    new_bags: list[tuple[str, ...]],
) -> None:
    changed = True
    while changed:
        changed = False
        for node in list(active_nodes):
            neighbors_in = [n for n in adjacency[node] if n in active_nodes]
            if len(neighbors_in) == 1:
                neighbor = neighbors_in[0]
                if set(new_bags[node]).issubset(set(new_bags[neighbor])):
                    active_nodes.discard(node)
                    changed = True


def restrict(td: TreeDecomposition, subset: frozenset[str]) -> TreeDecomposition:
    """Return the decomposition obtained by replacing every bag B_t with
    B_t ∩ S, then applying the documented deterministic cleanup of empty/
    redundant tree nodes. Bind the result to the induced source graph G[S]."""
    if not subset.issubset(set(td.graph.vertices)):
        raise ValueError("subset must contain only declared source vertices")
    # New source graph induced by S.
    keep_list = [v for v in td.graph.vertices if v in subset]
    new_edges = [(a, b) for a, b in td.graph.edges if a in subset and b in subset]
    new_graph = SimpleUndirectedGraph(
        vertices=tuple(keep_list),
        edges=tuple(new_edges),
    )
    # New bags intersected with S.
    new_bags = [tuple(v for v in bag if v in subset) for bag in td.bags]
    # Cleanup: remove empty bags, contracting through deleted internal nodes
    # to keep the tree connected.
    keep_indices = [i for i, bag in enumerate(new_bags) if bag]
    active_nodes = set(keep_indices)
    int_edges = _int_edges(td)
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(td.tree_nodes))}
    for a, b in int_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    # Build edges by contracting through deleted nodes: for each pair of
    # active nodes, check if there's a path through deleted nodes.
    new_edges_list: list[tuple[int, int]] = []
    visited_edges: set[tuple[int, int]] = set()
    for i in keep_indices:
        # Traverse from node i through deleted nodes to find all active neighbors.
        stack: list[tuple[int, set[int]]] = [(i, set())]
        while stack:
            node, path = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in active_nodes and neighbor != i:
                    edge = (min(i, neighbor), max(i, neighbor))
                    if edge not in visited_edges:
                        new_edges_list.append(edge)
                        visited_edges.add(edge)
                elif neighbor not in active_nodes and neighbor not in path:
                    stack.append((neighbor, path | {node}))
    # Prune redundant leaves: leaf whose bag is a subset of its neighbor.
    # Build adjacency from new_edges_list for the active subgraph
    active_adj: dict[int, list[int]] = {i: [] for i in active_nodes}
    for a, b in new_edges_list:
        active_adj[a].append(b)
        active_adj[b].append(a)
    _prune_redundant_leaves(active_nodes, active_adj, new_bags)
    final_nodes = sorted(active_nodes)
    result_nodes = tuple(td.tree_nodes[i] for i in final_nodes)
    result_edges = []
    for a, b in new_edges_list:
        if a in active_nodes and b in active_nodes:
            la, lb = td.tree_nodes[a], td.tree_nodes[b]
            result_edges.append((la, lb) if la <= lb else (lb, la))
    result_edges = sorted(set(result_edges))
    result_bags = tuple(new_bags[i] for i in final_nodes)
    return TreeDecomposition(
        graph=new_graph,
        tree_nodes=result_nodes,
        tree_edges=tuple(result_edges),
        bags=result_bags,
    )


def bag_intersection_graph(td: TreeDecomposition) -> BagIntersectionGraphResult:
    """Return the weighted tree itself with each edge labelled by its exact
    adhesion set/size and each node labelled by bag size."""
    result = adhesions(td)
    node_labels: list[BagNode] = []
    for i, bag in enumerate(td.bags):
        node_labels.append(BagNode(node=td.tree_nodes[i], bag_size=len(bag)))
    return BagIntersectionGraphResult(
        nodes=tuple(node_labels),
        edges=result.edges,
        max_adhesion=result.max_adhesion,
    )
