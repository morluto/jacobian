"""Tests for 3-term progression hypergraph construction."""

from itertools import product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics import progression_hypergraph
from jacobian.math.combinatorics._progression_hypergraph import (
    PROGRESSION_HYPERGRAPH_OPERATION,
)
from jacobian.math.combinatorics._progression_hypergraph_models import (
    MAX_GROUP_ORDER,
    ProgressionHypergraphRequest,
    ProgressionHypergraphResult,
    progression_edge_bound,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import parameters
from jacobian.math.groups.finite_abelian import FiniteAbelianProductGroup


def _group(*moduli: int) -> FiniteAbelianProductGroup:
    return FiniteAbelianProductGroup(moduli=moduli)


def _exhaustive_3term_edge_count(moduli: tuple[int, ...]) -> int:
    """Independently enumerate unordered nondegenerate 3-AP vertex sets."""

    elements = tuple(product(*(range(modulus) for modulus in moduli)))
    edges = {
        frozenset(
            tuple(
                (start[axis] + multiple * step[axis]) % moduli[axis]
                for axis in range(len(moduli))
            )
            for multiple in range(3)
        )
        for start in elements
        for step in elements
    }
    return sum(len(edge) == 3 for edge in edges)


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


@pytest.mark.parametrize(
    "moduli",
    ((2, 2), (2, 3), (3, 3), (4, 3), (5, 5), (2, 2, 3), (4, 4)),
)
def test_product_group_edge_bound_matches_independent_enumeration(
    moduli: tuple[int, ...],
) -> None:
    """Mixed 2- and 3-torsion groups distinguish the edge multiplicities."""

    assert progression_edge_bound(_group(*moduli)) == _exhaustive_3term_edge_count(
        moduli
    )


def test_elementary_two_group_at_vertex_boundary_is_admitted_and_composable() -> None:
    """The exact empty-edge output, rather than a dense proxy, controls admission."""

    request = ProgressionHypergraphRequest(group=_group(*(2,) * 8))
    result = PROGRESSION_HYPERGRAPH_OPERATION.run(request)

    assert len(result.vertex_elements) == MAX_GROUP_ORDER
    assert result.hypergraph.edges == ()
    decoded = ProgressionHypergraphResult.model_validate_json(result.model_dump_json())
    profile = parameters(decoded.hypergraph)
    assert profile.vertex_count == MAX_GROUP_ORDER
    assert profile.edge_count == 0
    assert profile.total_incidences == 0


def test_product_group_over_sound_edge_bound_is_rejected() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        progression_hypergraph(_group(13, 13))
