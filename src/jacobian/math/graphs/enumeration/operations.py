"""Atlas generation followed by exact coloring orbits under automorphisms."""

from itertools import product
from math import factorial
from typing import Any

import networkx as nx

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.enumeration._models import (
    ConnectedColoredGraphFamily,
    ConnectedColoredGraphsRequest,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def connected_colored_graphs(
    source: ConnectedColoredGraphsRequest,
) -> ConnectedColoredGraphFamily:
    """Enumerate complete colored families from the maintained seven-vertex atlas.

    Uncolored graphs are already isomorph-free. Two colorings of one such
    graph are equivalent exactly when an automorphism transports them. Choose
    the lexicographically first color vector in each orbit; fixed vertex labels
    are v00,... in the atlas representative. No factorial relabeling is rerun
    for each coloring, and colors themselves are never renamed.
    """
    request_checkpoint("before colored graph enumeration")
    costs = {entry.colors: entry.cost for entry in source.edge_costs}
    minimum_cost = min(costs.values(), default=source.cost_bound + 1)
    graphs: list[Any] = [
        g
        for g in nx.graph_atlas_g()
        if 2 <= len(g) <= source.vertex_bound
        and g.number_of_edges() * minimum_cost <= source.cost_bound
        and nx.is_connected(g)
    ]
    # VF2 explores at most sum_k n!/(n-k)! < 3*n! partial injections;
    # n^3 bounds feasibility bookkeeping. Orbit work is at most c^n*n!*n.
    work = sum(
        factorial(len(g)) * (3 * len(g) ** 3 + len(source.palette) ** len(g) * len(g))
        for g in graphs
    )
    candidates = sum(len(source.palette) ** len(g) for g in graphs)
    if work > 150_000_000 or candidates > 20_000:
        raise OperationResourceAdmissionError(
            location=("vertex_bound", "palette", "cost_bound"),
            code="graph.colored_enumeration_work",
            message="complete colored enumeration exceeds 150000000 search units or 20000 candidate colorings",
        )
    results = []
    for graph in graphs:
        request_checkpoint("during colored graph enumeration")
        n = len(graph)
        permutations = tuple(
            tuple(mapping[i] for i in range(n))
            for mapping in nx.algorithms.isomorphism.GraphMatcher(
                graph, graph
            ).isomorphisms_iter()
        )
        labels = tuple(f"v{i:02d}" for i in range(n))
        edges = tuple(
            sorted(
                (min(labels[a], labels[b]), max(labels[a], labels[b]))
                for a, b in graph.edges
            )
        )
        seen = set()
        for colors in product(source.palette, repeat=n):
            request_checkpoint("during coloring orbit enumeration")
            if colors in seen:
                continue
            total = 0
            for a, b in graph.edges:
                cost = costs.get(tuple(sorted((colors[a], colors[b]))))
                if cost is None:
                    break
                total += cost
            else:
                if total <= source.cost_bound:
                    orbit = {
                        tuple(colors[p[i]] for i in range(n)) for p in permutations
                    }
                    seen.update(orbit)
                    canonical_colors = min(orbit)
                    results.append(
                        ColoredUndirectedGraph(
                            graph=SimpleUndirectedGraph(vertices=labels, edges=edges),
                            vertex_colors=canonical_colors,
                        )
                    )
    results.sort(key=lambda g: (len(g.graph.vertices), g.graph.edges, g.vertex_colors))
    return ConnectedColoredGraphFamily(**source.model_dump(), graphs=tuple(results))


__all__ = ["connected_colored_graphs"]
