"""Free-tree enumeration kernel."""

from __future__ import annotations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.tree_enumeration.free_tree_enumeration._models import (
    MAX_ORDER,
    FreeTreeEnumerationResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["enumerate_free_trees"]


def _canonical_edge(left: int, right: int) -> tuple[str, str]:
    first, second = str(left), str(right)
    return (first, second) if first < second else (second, first)


def enumerate_free_trees(order: int) -> FreeTreeEnumerationResult:
    """Return one canonical SimpleUndirectedGraph for every isomorphism
    class of free trees on *order* vertices.

    Uses NetworkX's maintained WROM implementation.
    """
    if type(order) is not int or not 0 <= order <= MAX_ORDER:
        raise OperationDomainValidationError(
            location=("order",),
            code="graph.free_tree_enumeration_order_out_of_range",
            message=f"free-tree enumeration supports orders from 0 through {MAX_ORDER}",
        )
    if order == 0:
        return FreeTreeEnumerationResult(order=0, trees=(), count=0)
    if order == 1:
        singleton = SimpleUndirectedGraph(vertices=("0",), edges=())
        return FreeTreeEnumerationResult(order=1, trees=(singleton,), count=1)

    graphs: list[SimpleUndirectedGraph] = []
    for nx_tree in nx.nonisomorphic_trees(order):
        tree_graph: nx.Graph[int] = nx.Graph(nx_tree)
        vertices = tuple(str(i) for i in range(order))
        edges = tuple(_canonical_edge(u, v) for u, v in tree_graph.edges())
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
