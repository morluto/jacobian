"""Free-tree enumeration kernel."""

from __future__ import annotations

import networkx as nx

from jacobian.math.graphs.tree_enumeration.free_tree_enumeration._models import (
    FreeTreeEnumerationResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["enumerate_free_trees"]


def enumerate_free_trees(order: int) -> FreeTreeEnumerationResult:
    """Return one canonical SimpleUndirectedGraph for every isomorphism
    class of free trees on *order* vertices.

    Uses NetworkX's maintained WROM implementation.
    """
    if order == 0:
        return FreeTreeEnumerationResult(order=0, trees=(), count=0)

    graphs: list[SimpleUndirectedGraph] = []
    for nx_tree in nx.nonisomorphic_trees(order):
        vertices = tuple(str(i) for i in range(order))
        edges = tuple(
            tuple(sorted((str(u), str(v))))
            for u, v in nx_tree.edges()
        )
        graphs.append(
            SimpleUndirectedGraph(
                vertices=vertices,
                edges=edges,
            )
        )

    return FreeTreeEnumerationResult(
        order=order,
        trees=tuple(graphs),
        count=len(graphs),
    )
