"""Tests for 3-term progression hypergraph construction."""

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics import progression_hypergraph
from jacobian.math.combinatorics._progression_hypergraph_models import (
    MAX_GROUP_ORDER,
    ProgressionHypergraphRequest,
)


def test_z3() -> None:
    """Z/3Z has 3 vertices. The only 3-AP is {0,1,2}."""
    result = progression_hypergraph(3)
    assert len(result.hypergraph.vertices) == 3
    assert len(result.hypergraph.edges) == 1


def test_z5() -> None:
    """Z/5Z has 5 vertices and 10 edges (C(5,3) = 10, all triples are 3-APs)."""
    result = progression_hypergraph(5)
    assert len(result.hypergraph.vertices) == 5
    assert len(result.hypergraph.edges) == 10


def test_z7_count() -> None:
    """Z/7Z: n*(n-1)/2 = 21 edges."""
    result = progression_hypergraph(7)
    assert len(result.hypergraph.edges) == 21


def test_all_edges_are_3_uniform() -> None:
    """Every edge should have exactly 3 distinct vertices."""
    result = progression_hypergraph(7)
    for _, members in result.hypergraph.edges:
        assert len(members) == 3
        assert len(set(members)) == 3


def test_maximum_admitted_order_fits_hypergraph_representation() -> None:
    """The request ceiling admits the largest representable cyclic group."""
    result = progression_hypergraph(MAX_GROUP_ORDER)
    assert MAX_GROUP_ORDER == 156
    assert len(result.hypergraph.edges) == 11_908
    assert 3 * len(result.hypergraph.edges) == 35_724


def test_first_order_beyond_representation_ceiling_is_rejected() -> None:
    """The former public maximum would overflow FiniteHypergraph at 157."""
    with pytest.raises(ValidationError) as exc_info:
        ProgressionHypergraphRequest(group_order=MAX_GROUP_ORDER + 1)

    assert exc_info.value.errors()[0]["type"] == "less_than_equal"
