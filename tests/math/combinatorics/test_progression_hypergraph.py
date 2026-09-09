"""Tests for 3-term progression hypergraph construction."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics import progression_hypergraph
from jacobian.math.combinatorics._progression_hypergraph_models import (
    MAX_GROUP_ORDER,
)
from jacobian.math.groups.finite_abelian import FiniteAbelianProductGroup


def _group(*moduli: int) -> FiniteAbelianProductGroup:
    return FiniteAbelianProductGroup(moduli=moduli)


def test_z3() -> None:
    """Z/3Z has 3 vertices. The only 3-AP is {0,1,2}."""
    result = progression_hypergraph(_group(3))
    assert len(result.hypergraph.vertices) == 3
    assert len(result.hypergraph.edges) == 1


def test_z5() -> None:
    """Z/5Z has 5 vertices and 10 edges (C(5,3) = 10, all triples are 3-APs)."""
    result = progression_hypergraph(_group(5))
    assert len(result.hypergraph.vertices) == 5
    assert len(result.hypergraph.edges) == 10


def test_z7_count() -> None:
    """Z/7Z: n*(n-1)/2 = 21 edges."""
    result = progression_hypergraph(_group(7))
    assert len(result.hypergraph.edges) == 21


def test_all_edges_are_3_uniform() -> None:
    """Every edge should have exactly 3 distinct vertices."""
    result = progression_hypergraph(_group(7))
    for _, members in result.hypergraph.edges:
        assert len(members) == 3
        assert len(set(members)) == 3


def test_maximum_admitted_order_fits_hypergraph_representation() -> None:
    """The request ceiling admits the largest representable cyclic group."""
    result = progression_hypergraph(_group(156))
    assert MAX_GROUP_ORDER == 256
    assert len(result.hypergraph.edges) == 11_908
    assert 3 * len(result.hypergraph.edges) == 35_724


def test_product_group_progressions_are_complete() -> None:
    result = progression_hypergraph(_group(3, 3))
    assert len(result.hypergraph.vertices) == 9
    assert len(result.hypergraph.edges) == 12
    binding = {row.vertex: row.element.coordinates for row in result.vertex_elements}
    for _, vertices in result.hypergraph.edges:
        points = [binding[vertex] for vertex in vertices]
        assert any(
            all(
                (points[left][axis] + points[right][axis]) % 3
                == 2 * points[middle][axis] % 3
                for axis in range(2)
            )
            for middle, left, right in ((0, 1, 2), (1, 0, 2), (2, 0, 1))
        )


def test_product_group_over_sound_edge_bound_is_rejected() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        progression_hypergraph(_group(13, 13))
