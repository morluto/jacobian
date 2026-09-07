"""Declared action admission permits large sparse-action orbit partitions."""

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.symmetry._models import (
    GraphAutomorphismGenerator,
    GraphSymmetryOrbitResult,
)
from jacobian.math.graphs.symmetry.operations import graph_symmetry_orbits
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def _graph(order: int) -> ColoredUndirectedGraph:
    vertices = tuple(sorted(map(str, range(order))))
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=vertices, edges=tuple(combinations(vertices, 2))
        )
    )


def test_k92_trivial_action_has_singleton_orbits_above_old_edge_cap() -> None:
    graph = _graph(92)
    result = graph_symmetry_orbits(graph, ())
    assert result.edge_orbit_count == 4186
    assert all(orbit.members == (orbit.representative,) for orbit in result.edge_orbits)
    assert (
        GraphSymmetryOrbitResult.model_validate_json(result.model_dump_json()) == result
    )


def test_large_graph_transposition_orbits_match_direct_action() -> None:
    graph = _graph(100)
    a, b, *others = graph.graph.vertices
    generator = GraphAutomorphismGenerator(
        generator_id="swap",
        mapping=tuple(
            (v, b if v == a else a if v == b else v) for v in graph.graph.vertices
        ),
    )
    result = graph_symmetry_orbits(graph, (generator,))
    assert {frozenset(orbit.members) for orbit in result.vertex_orbits} == {
        frozenset((a, b)),
        *(frozenset((v,)) for v in others),
    }
    assert sum(len(orbit.members) == 2 for orbit in result.edge_orbits) == 98


def test_large_action_table_retains_old_memory_envelope() -> None:
    graph = _graph(256)
    generators = tuple(
        GraphAutomorphismGenerator(
            generator_id=f"g{i}", mapping=tuple((v, v) for v in graph.graph.vertices)
        )
        for i in range(64)
    )
    with pytest.raises(OperationResourceAdmissionError, match="action entries"):
        graph_symmetry_orbits(graph, generators)
