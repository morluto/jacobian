"""Monochromatic clique hypergraph constructor."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.monochromatic_clique._models import (
    MonochromaticCliqueHypergraphResult,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

__all__ = ["construct_monochromatic_clique_hypergraph"]


def construct_monochromatic_clique_hypergraph(
    colored_graph: ColoredUndirectedGraph,
    clique_order: int,
) -> MonochromaticCliqueHypergraphResult:
    """Construct the t-uniform monochromatic-clique hypergraph.

    For each t-element vertex subset, check whether all C(t,2) edges
    share the same colour. If so, the subset is a monochromatic t-clique
    and becomes a hyperedge.
    """
    graph = colored_graph.graph
    vertices = list(graph.vertices)
    edges = list(graph.edges)
    edge_colors = colored_graph.edge_colors

    edge_to_color: dict[tuple[str, str], str] = {}
    for i, (a, b) in enumerate(edges):
        edge_to_color[(a, b)] = edge_colors[i]

    hyper_edges: list[tuple[str, tuple[str, ...]]] = []
    edge_index = 0

    for subset in combinations(vertices, clique_order):
        colors = set()
        is_mono = True
        for i in range(clique_order):
            for j in range(i + 1, clique_order):
                a, b = subset[i], subset[j]
                if (a, b) in edge_to_color:
                    colors.add(edge_to_color[(a, b)])
                elif (b, a) in edge_to_color:
                    colors.add(edge_to_color[(b, a)])
                else:
                    is_mono = False
                    break
            if not is_mono:
                break
        if is_mono and len(colors) == 1:
            hyper_edges.append((f"clique_{edge_index}", tuple(sorted(subset))))
            edge_index += 1

    hypergraph = FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple(hyper_edges),
    )
    return MonochromaticCliqueHypergraphResult(
        colored_graph=colored_graph,
        clique_order=clique_order,
        hypergraph=hypergraph,
    )
