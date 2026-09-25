"""Bounded graph-to-flag-complex construction."""

from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.topology.release import graph_clique_complex


def test_graph_clique_complex_matches_independent_powerset_oracle() -> None:
    for vertex_count in range(1, 5):
        possible_edges = tuple(combinations(range(vertex_count), 2))
        for mask in range(1 << len(possible_edges)):
            edges = tuple(
                edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
            )
            graph = IndexedSimpleUndirectedGraph(vertex_count=vertex_count, edges=edges)
            result = graph_clique_complex(graph)
            edge_set = set(edges)
            cliques = {
                subset
                for size in range(1, vertex_count + 1)
                for subset in combinations(range(vertex_count), size)
                if all(
                    (left, right) in edge_set for left, right in combinations(subset, 2)
                )
            }
            maximal = tuple(
                sorted(
                    tuple(f"v{vertex}" for vertex in subset)
                    for subset in cliques
                    if not any(set(subset) < set(other) for other in cliques)
                )
            )
            assert set(result.clique_facets) == set(maximal)
            assert result.clique_complex.closure_size == len(cliques)


def test_graph_clique_complex_boundary_and_oversized_request() -> None:
    full = IndexedSimpleUndirectedGraph(
        vertex_count=8,
        edges=tuple((left, right) for left in range(8) for right in range(left + 1, 8)),
    )
    assert graph_clique_complex(full).clique_complex.closure_size == 255

    oversized = IndexedSimpleUndirectedGraph(vertex_count=9, edges=())
    with pytest.raises(OperationResourceAdmissionError, match="between 1 and 8"):
        graph_clique_complex(oversized)
