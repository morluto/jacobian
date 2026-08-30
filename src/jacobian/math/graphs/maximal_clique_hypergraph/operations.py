"""Maximal-clique hypergraph kernel."""

from __future__ import annotations

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphResult,
    _maximal_clique_admission_error,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["construct_maximal_clique_hypergraph"]


def construct_maximal_clique_hypergraph(
    graph: SimpleUndirectedGraph,
) -> MaximalCliqueHypergraphResult:
    """Construct the hypergraph whose edges are the nontrivial maximal cliques.

    Uses NetworkX's Bron-Kerbosch with pivoting to find all maximal cliques.
    Only cliques of size >= 2 are retained (isolated vertices induce no edge).
    """
    failure = _maximal_clique_admission_error(graph)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("graph",), code=f"maximal_clique.{code}", message=message
        )
    nx_graph: nx.Graph[str] = nx.Graph()
    for v in graph.vertices:
        nx_graph.add_node(v)
    for u, v in graph.edges:
        nx_graph.add_edge(u, v)

    cliques = [c for c in nx.find_cliques(nx_graph) if len(c) >= 2]

    # Sort cliques for deterministic output
    cliques.sort(key=lambda c: tuple(sorted(c)))

    edges = tuple((f"clique_{i}", tuple(sorted(c))) for i, c in enumerate(cliques))

    hypergraph = FiniteHypergraph(
        vertices=graph.vertices,
        edges=edges,
    )

    return MaximalCliqueHypergraphResult(
        graph=graph,
        hypergraph=hypergraph,
        clique_count=len(cliques),
    )
