"""Provider-independent values for exact tree-decomposition operations.

A *tree decomposition* of a finite simple undirected graph ``G`` is a pair
``(T, B)`` where ``T`` is a tree whose vertices (tree nodes) carry finite
*bags* ``B_t`` — subsets of ``G``'s vertices — such that:

1. the decomposition graph is a tree;
2. every source vertex occurs in at least one bag;
3. every source edge has both endpoints in at least one bag; and
4. for each source vertex, the tree nodes whose bags contain it form a
   connected subtree (the *connectedness* axiom).

These are value-construction invariants. They are **not** exposed as a
public ``.check`` operation.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_TREE_NODES = 256
MAX_BAG_SIZE = 64


def _is_tree(node_count: int, edges: list[tuple[int, int]]) -> bool:
    if node_count == 0:
        return False
    if len(edges) != node_count - 1:
        return False
    if node_count == 1:
        return True
    parent = list(range(node_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


def _validate_bags(bags: tuple[tuple[str, ...], ...], vertex_set: set[str]) -> None:
    for bag in bags:
        if list(bag) != sorted(bag):
            raise PydanticCustomError(
                "graph.bag_members_must_be_a_sorted_tuple",
                "bag members must be a sorted tuple",
            )
        if len(set(bag)) != len(bag):
            raise PydanticCustomError(
                "graph.bag_members_must_be_unique", "bag members must be unique"
            )
        if len(bag) > MAX_BAG_SIZE:
            raise PydanticCustomError(
                "graph.bag_size_exceeds_the_bounded_budget",
                "bag size exceeds the bounded budget",
            )
        for vertex in bag:
            if vertex not in vertex_set:
                raise PydanticCustomError(
                    "graph.bag_references_an_undeclared_graph_vertex",
                    "bag references an undeclared graph vertex",
                )


def _check_vertex_coverage(
    bags: tuple[tuple[str, ...], ...], vertex_set: set[str]
) -> None:
    all_bag_vertices: set[str] = set()
    for bag in bags:
        all_bag_vertices.update(bag)
    if all_bag_vertices != vertex_set:
        raise PydanticCustomError(
            "graph.every_source_vertex_must_occur_in_at_least_one_b",
            "every source vertex must occur in at least one bag",
        )


def _check_edge_coverage(
    graph: SimpleUndirectedGraph, bags: tuple[tuple[str, ...], ...]
) -> None:
    for left, right in graph.edges:
        if not any(left in bag and right in bag for bag in bags):
            raise PydanticCustomError(
                "graph.every_source_edge_have_both_endpoints_at",
                "every source edge must have both endpoints in at least one bag",
            )


def _check_connectedness(
    graph: SimpleUndirectedGraph,
    bags: tuple[tuple[str, ...], ...],
    int_edges: list[tuple[int, int]],
    node_count: int,
) -> None:
    adjacency: dict[int, list[int]] = {i: [] for i in range(node_count)}
    for a, b in int_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    for vertex in graph.vertices:
        containing = [i for i, bag in enumerate(bags) if vertex in bag]
        if not containing:
            continue
        reached = {containing[0]}
        stack = [containing[0]]
        while stack:
            current = stack.pop()
            for nxt in adjacency[current]:
                if nxt in containing and nxt not in reached:
                    reached.add(nxt)
                    stack.append(nxt)
        if reached != set(containing):
            raise PydanticCustomError(
                "graph.connectedness_axiom_violated_for_vertex_vertex",
                "the connectedness axiom is violated for vertex " + vertex,
            )


class TreeDecomposition(StrictModel):
    """An immutable well-formed tree decomposition of a source graph."""

    graph: SimpleUndirectedGraph
    tree_nodes: tuple[str, ...] = Field(min_length=1, max_length=MAX_TREE_NODES)
    tree_edges: tuple[tuple[str, str], ...] = Field(max_length=MAX_TREE_NODES)
    bags: tuple[tuple[str, ...], ...]

    @model_validator(mode="after")
    def require_well_formed(self) -> Self:
        if len(self.bags) != len(self.tree_nodes):
            raise PydanticCustomError(
                "graph.bags_must_have_one_entry_per_tree_node",
                "bags must have one entry per tree node",
            )
        if len(set(self.tree_nodes)) != len(self.tree_nodes):
            raise PydanticCustomError(
                "graph.tree_nodes_must_be_unique", "tree_nodes must be unique"
            )
        node_set = set(self.tree_nodes)
        for left, right in self.tree_edges:
            if left == right:
                raise PydanticCustomError(
                    "graph.tree_edges_must_not_be_loops", "tree edges must not be loops"
                )
            if left not in node_set or right not in node_set:
                raise PydanticCustomError(
                    "graph.tree_edge_references_an_undeclared_node",
                    "tree edge references an undeclared node",
                )
        edge_pairs = [(a, b) if a <= b else (b, a) for a, b in self.tree_edges]
        if len(set(edge_pairs)) != len(edge_pairs):
            raise PydanticCustomError(
                "graph.tree_edges_must_be_unique", "tree edges must be unique"
            )
        index_of = {label: i for i, label in enumerate(self.tree_nodes)}
        int_edges = [(index_of[a], index_of[b]) for a, b in edge_pairs]
        if not _is_tree(len(self.tree_nodes), int_edges):
            raise PydanticCustomError(
                "graph.tree_edges_do_not_form_a_tree", "tree edges do not form a tree"
            )
        vertex_set = set(self.graph.vertices)
        _validate_bags(self.bags, vertex_set)
        _check_vertex_coverage(self.bags, vertex_set)
        _check_edge_coverage(self.graph, self.bags)
        _check_connectedness(self.graph, self.bags, int_edges, len(self.tree_nodes))
        return self


__all__ = [
    "MAX_BAG_SIZE",
    "MAX_TREE_NODES",
    "TreeDecomposition",
]
