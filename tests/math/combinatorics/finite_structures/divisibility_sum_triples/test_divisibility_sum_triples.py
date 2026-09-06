from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.finite_structures.divisibility_sum_triples._models import (
    DivisibilitySumTriplesRequest,
)
from jacobian.math.combinatorics.finite_structures.divisibility_sum_triples.operations import (
    construct_divisibility_sum_triples_hypergraph,
    verify_divisibility_sum_triples,
)


def _edge_member_sets(result):
    return [frozenset(members) for _, members in result.hypergraph.edges]


def _sorted_ints(members):
    return sorted(int(x) for x in members)


def test_fixture_1_to_4() -> None:
    """On [1,4], triples with a<b<c and a dividing b+c include {1,2,3}, {1,2,4}, and {1,3,4}."""
    result = construct_divisibility_sum_triples_hypergraph(1, 4)
    edges = _edge_member_sets(result)
    assert frozenset({"1", "2", "3"}) in edges
    assert frozenset({"1", "2", "4"}) in edges
    assert frozenset({"1", "3", "4"}) in edges


def test_replay_divisibility() -> None:
    """Every edge satisfies a | (b + c)."""
    result = construct_divisibility_sum_triples_hypergraph(1, 10)
    for _, members in result.hypergraph.edges:
        vals = _sorted_ints(members)
        a, b, c = vals
        assert a < b < c
        assert a != 0
        assert (b + c) % a == 0


def test_replay_ordering() -> None:
    """Every edge is in strictly increasing numerical order."""
    result = construct_divisibility_sum_triples_hypergraph(1, 10)
    for _, members in result.hypergraph.edges:
        vals = _sorted_ints(members)
        assert vals[0] < vals[1] < vals[2]


def test_empty_interval() -> None:
    """L > U is rejected."""
    with pytest.raises(ValidationError):
        DivisibilitySumTriplesRequest(lower_bound=5, upper_bound=3)


def test_short_interval() -> None:
    """Interval too short for any triple has no edges."""
    result = construct_divisibility_sum_triples_hypergraph(1, 2)
    assert len(result.hypergraph.edges) == 0


def test_vertex_count() -> None:
    """Vertices are all integers in the interval."""
    result = construct_divisibility_sum_triples_hypergraph(1, 5)
    assert len(result.hypergraph.vertices) == 5


def test_exhaustive_comparison() -> None:
    """Compare against independent brute-force enumeration."""
    lo, hi = 1, 6
    result = construct_divisibility_sum_triples_hypergraph(lo, hi)
    expected = set()
    for a in range(lo, hi + 1):
        for b in range(a + 1, hi + 1):
            for c in range(b + 1, hi + 1):
                if a != 0 and (b + c) % a == 0:
                    expected.add(frozenset([str(a), str(b), str(c)]))
    actual = {frozenset(members) for _, members in result.hypergraph.edges}
    assert actual == expected


def test_no_duplicates() -> None:
    """Each triple appears exactly once."""
    result = construct_divisibility_sum_triples_hypergraph(1, 8)
    edges = _edge_member_sets(result)
    assert len(edges) == len(set(edges))


def test_result_preserves_bounds() -> None:
    """Result retains source bounds."""
    result = construct_divisibility_sum_triples_hypergraph(3, 8)
    assert result.lower_bound == 3
    assert result.upper_bound == 8


def test_result_sensitive_admission_accepts_edge_free_shifted_interval() -> None:
    result = construct_divisibility_sum_triples_hypergraph(1000, 1042)
    assert result.hypergraph.edges == ()


def test_rejects_actual_hypergraph_envelope() -> None:
    with pytest.raises(ValueError, match="exact triple family"):
        construct_divisibility_sum_triples_hypergraph(1, 90)


def test_native_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        construct_divisibility_sum_triples_hypergraph(4, 1)


def test_serialized_divisibility_claim_is_verifiable_and_forgery_is_structural() -> (
    None
):
    result = construct_divisibility_sum_triples_hypergraph(1, 4)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_divisibility_sum_triples(decoded)

    forged = result.model_dump(mode="json")
    forged["hypergraph"]["edges"][0][1] = ["1", "2", "4"]
    forged_decoded = type(result).model_validate_json(json.dumps(forged))
    assert not verify_divisibility_sum_triples(forged_decoded)
