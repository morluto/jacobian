"""Comprehensive tests for the k-term arithmetic-progression hypergraph constructor."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.arithmetic_progression_hypergraph._models import (
    MAX_INTERVAL_SIZE,
    _edge_count,
)
from jacobian.math.combinatorics.arithmetic_progression_hypergraph.operations import (
    construct_arithmetic_progression_hypergraph,
)

# ---------------------------------------------------------------------------
# Edge-count formula
# ---------------------------------------------------------------------------


def test_edge_count_formula_k3_n1() -> None:
    assert _edge_count(1, 3) == 0


def test_edge_count_formula_k3_n2() -> None:
    assert _edge_count(2, 3) == 0


def test_edge_count_formula_k3_n3() -> None:
    assert _edge_count(3, 3) == 1


def test_edge_count_formula_k3_n4() -> None:
    assert _edge_count(4, 3) == 2


def test_edge_count_formula_k3_n5() -> None:
    assert _edge_count(5, 3) == 4


def test_edge_count_formula_k4_n4() -> None:
    assert _edge_count(4, 4) == 1


def test_edge_count_formula_k4_n5() -> None:
    assert _edge_count(5, 4) == 2


def test_edge_count_formula_k4_n6() -> None:
    assert _edge_count(6, 4) == 3


def test_edge_count_formula_k3_n212() -> None:
    """The issue's source: N=212, k=3 should give 11,130 edges."""
    assert _edge_count(212, 3) == 11130
    assert 3 * _edge_count(212, 3) == 33390


# ---------------------------------------------------------------------------
# Empty and degenerate intervals
# ---------------------------------------------------------------------------


def test_n1_singleton_interval_k3() -> None:
    """A single-vertex interval [0,0] with k=3 has no edges."""
    result = construct_arithmetic_progression_hypergraph(0, 0, 3)
    assert result.hypergraph.vertices == ("0",)
    assert result.hypergraph.edges == ()


def test_n2_interval_k3() -> None:
    """Interval [1,2] with k=3 has no edges (too few vertices)."""
    result = construct_arithmetic_progression_hypergraph(1, 2, 3)
    assert result.hypergraph.vertices == ("1", "2")
    assert result.hypergraph.edges == ()


def test_n3_interval_k3() -> None:
    """Interval [1,3] with k=3 has exactly 1 edge: (1,2,3)."""
    result = construct_arithmetic_progression_hypergraph(1, 3, 3)
    assert len(result.hypergraph.edges) == 1
    eid, members = result.hypergraph.edges[0]
    assert members == ("1", "2", "3")
    assert eid == "(1,1)"


def test_n4_interval_k3() -> None:
    """Interval [1,4] with k=3 has exactly 2 edges."""
    result = construct_arithmetic_progression_hypergraph(1, 4, 3)
    assert len(result.hypergraph.edges) == 2
    edge_sets = {frozenset(members) for _, members in result.hypergraph.edges}
    assert edge_sets == {
        frozenset({"1", "2", "3"}),
        frozenset({"2", "3", "4"}),
    }


# ---------------------------------------------------------------------------
# Vertex correctness
# ---------------------------------------------------------------------------


def test_vertices_are_canonical_interval_strings() -> None:
    result = construct_arithmetic_progression_hypergraph(-3, 3, 3)
    assert result.hypergraph.vertices == ("-3", "-2", "-1", "0", "1", "2", "3")


def test_vertex_count_matches_interval() -> None:
    result = construct_arithmetic_progression_hypergraph(10, 20, 3)
    assert len(result.hypergraph.vertices) == 11


# ---------------------------------------------------------------------------
# Edge-count verification against formula
# ---------------------------------------------------------------------------


def test_edge_count_matches_formula_n5_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 3)
    assert len(result.hypergraph.edges) == _edge_count(5, 3)
    assert len(result.hypergraph.edges) == 4


def test_edge_count_matches_formula_n10_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(0, 9, 3)
    assert len(result.hypergraph.edges) == _edge_count(10, 3)


def test_edge_count_matches_formula_n10_k4() -> None:
    result = construct_arithmetic_progression_hypergraph(0, 9, 4)
    assert len(result.hypergraph.edges) == _edge_count(10, 4)


def test_edge_count_matches_formula_n20_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(5, 24, 3)
    assert len(result.hypergraph.edges) == _edge_count(20, 3)


def test_edge_count_matches_formula_n20_k5() -> None:
    result = construct_arithmetic_progression_hypergraph(5, 24, 5)
    assert len(result.hypergraph.edges) == _edge_count(20, 5)


# ---------------------------------------------------------------------------
# Edge member correctness: every edge is a valid AP
# ---------------------------------------------------------------------------


def test_all_edges_are_valid_ap_n5_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 3)
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 3
        vals = sorted(int(m) for m in members)
        d = vals[1] - vals[0]
        assert d == vals[2] - vals[1]
        assert d > 0


def test_all_edges_are_valid_ap_n10_k4() -> None:
    result = construct_arithmetic_progression_hypergraph(0, 9, 4)
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 4
        vals = sorted(int(m) for m in members)
        d = vals[1] - vals[0]
        assert d == vals[2] - vals[1]
        assert d == vals[3] - vals[2]
        assert d > 0


def test_all_edges_are_valid_ap_n15_k5() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 15, 5)
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 5
        vals = sorted(int(m) for m in members)
        d = vals[1] - vals[0]
        for i in range(2, 5):
            assert d == vals[i] - vals[i - 1]
        assert d > 0


# ---------------------------------------------------------------------------
# All edges contained in interval
# ---------------------------------------------------------------------------


def test_all_edge_members_in_interval() -> None:
    lo, hi = 7, 30
    result = construct_arithmetic_progression_hypergraph(lo, hi, 3)
    for _eid, members in result.hypergraph.edges:
        for m in members:
            assert lo <= int(m) <= hi


def test_all_edge_members_in_interval_negative() -> None:
    lo, hi = -10, 5
    result = construct_arithmetic_progression_hypergraph(lo, hi, 3)
    for _eid, members in result.hypergraph.edges:
        for m in members:
            assert lo <= int(m) <= hi


# ---------------------------------------------------------------------------
# Edge ID format
# ---------------------------------------------------------------------------


def test_edge_ids_are_canonical_a_d_pairs() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 3)
    expected_ids = {"(1,1)", "(2,1)", "(3,1)", "(1,2)"}
    actual_ids = {eid for eid, _ in result.hypergraph.edges}
    assert actual_ids == expected_ids


def test_edge_id_encodes_first_term_and_difference() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 10, 3)
    for eid, members in result.hypergraph.edges:
        vals = sorted(int(m) for m in members)
        d = vals[1] - vals[0]
        a = vals[0]
        assert eid == f"({a},{d})"


# ---------------------------------------------------------------------------
# Edge uniqueness
# ---------------------------------------------------------------------------


def test_edge_ids_are_distinct() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 20, 3)
    ids = [eid for eid, _ in result.hypergraph.edges]
    assert len(ids) == len(set(ids))


def test_no_duplicate_edges() -> None:
    """No two edges should have the same vertex set."""
    result = construct_arithmetic_progression_hypergraph(1, 20, 3)
    edge_sets = [frozenset(members) for _, members in result.hypergraph.edges]
    assert len(edge_sets) == len(set(edge_sets))


# ---------------------------------------------------------------------------
# Uniformity
# ---------------------------------------------------------------------------


def test_uniformity_n5_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 3)
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 3


def test_uniformity_n10_k4() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 10, 4)
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 4


def test_uniformity_n15_k6() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 15, 6)
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 6


# ---------------------------------------------------------------------------
# Translation covariance
# ---------------------------------------------------------------------------


def test_translation_covariance() -> None:
    """Shifting [L,U] by c should produce the same edge structure."""
    r1 = construct_arithmetic_progression_hypergraph(1, 10, 3)
    r2 = construct_arithmetic_progression_hypergraph(100, 109, 3)
    assert len(r1.hypergraph.edges) == len(r2.hypergraph.edges)
    assert len(r1.hypergraph.vertices) == len(r2.hypergraph.vertices)


def test_translation_edge_count_matches() -> None:
    r1 = construct_arithmetic_progression_hypergraph(0, 15, 3)
    r2 = construct_arithmetic_progression_hypergraph(42, 57, 3)
    assert len(r1.hypergraph.edges) == len(r2.hypergraph.edges)


# ---------------------------------------------------------------------------
# k > N (no edges)
# ---------------------------------------------------------------------------


def test_k_exceeds_n_produces_no_edges() -> None:
    """When k > N, no k-term AP can fit, so edges should be empty."""
    result = construct_arithmetic_progression_hypergraph(1, 5, 10)
    assert result.hypergraph.edges == ()


def test_k_equals_n_plus_1() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 6)
    assert result.hypergraph.edges == ()


# ---------------------------------------------------------------------------
# Incidence count
# ---------------------------------------------------------------------------


def test_incidence_count_n5_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 3)
    total = sum(len(members) for _, members in result.hypergraph.edges)
    assert total == 3 * len(result.hypergraph.edges)


def test_incidence_count_n20_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 20, 3)
    total = sum(len(members) for _, members in result.hypergraph.edges)
    assert total == 3 * _edge_count(20, 3)


def test_incidence_count_n20_k4() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 20, 4)
    total = sum(len(members) for _, members in result.hypergraph.edges)
    assert total == 4 * _edge_count(20, 4)


# ---------------------------------------------------------------------------
# Bounded exhaustive: verify edge count matches formula for all N up to 30
# ---------------------------------------------------------------------------


def test_edge_count_matches_formula_exhaustive_k3() -> None:
    for n in range(1, 31):
        result = construct_arithmetic_progression_hypergraph(0, n - 1, 3)
        assert len(result.hypergraph.edges) == _edge_count(n, 3), (
            f"Edge count mismatch for N={n}, k=3"
        )


def test_edge_count_matches_formula_exhaustive_k4() -> None:
    for n in range(1, 31):
        result = construct_arithmetic_progression_hypergraph(0, n - 1, 4)
        assert len(result.hypergraph.edges) == _edge_count(n, 4), (
            f"Edge count mismatch for N={n}, k=4"
        )


def test_edge_count_matches_formula_exhaustive_k5() -> None:
    for n in range(1, 31):
        result = construct_arithmetic_progression_hypergraph(0, n - 1, 5)
        assert len(result.hypergraph.edges) == _edge_count(n, 5), (
            f"Edge count mismatch for N={n}, k=5"
        )


# ---------------------------------------------------------------------------
# Bounded exhaustive: verify all edges are valid APs for all N up to 20
# ---------------------------------------------------------------------------


def test_all_edges_are_valid_aps_exhaustive() -> None:
    for n in range(1, 21):
        for k in range(3, min(n + 1, 7)):
            result = construct_arithmetic_progression_hypergraph(0, n - 1, k)
            for _eid, members in result.hypergraph.edges:
                assert len(members) == k
                vals = sorted(int(m) for m in members)
                d = vals[1] - vals[0]
                assert d > 0
                for i in range(2, k):
                    assert vals[i] - vals[i - 1] == d
                assert all(0 <= v <= n - 1 for v in vals)


# ---------------------------------------------------------------------------
# Completeness: every AP in [L,U] is present as an edge
# ---------------------------------------------------------------------------


def test_completeness_n8_k3() -> None:
    """Every 3-term AP in [0,7] must appear as an edge."""
    lo, hi = 0, 7
    result = construct_arithmetic_progression_hypergraph(lo, hi, 3)
    edge_sets = {
        frozenset(int(m) for m in members) for _, members in result.hypergraph.edges
    }

    # Brute-force enumerate all 3-term APs
    expected_edges: set[frozenset[int]] = set()
    for a in range(lo, hi + 1):
        for d in range(1, hi - lo + 1):
            if a + 2 * d <= hi:
                expected_edges.add(frozenset({a, a + d, a + 2 * d}))

    assert edge_sets == expected_edges


def test_completeness_n8_k4() -> None:
    """Every 4-term AP in [0,7] must appear as an edge."""
    lo, hi = 0, 7
    result = construct_arithmetic_progression_hypergraph(lo, hi, 4)
    edge_sets = {
        frozenset(int(m) for m in members) for _, members in result.hypergraph.edges
    }

    expected_edges: set[frozenset[int]] = set()
    for a in range(lo, hi + 1):
        for d in range(1, hi - lo + 1):
            if a + 3 * d <= hi:
                expected_edges.add(frozenset({a, a + d, a + 2 * d, a + 3 * d}))

    assert edge_sets == expected_edges


def test_completeness_n10_k3() -> None:
    lo, hi = 0, 9
    result = construct_arithmetic_progression_hypergraph(lo, hi, 3)
    edge_sets = {
        frozenset(int(m) for m in members) for _, members in result.hypergraph.edges
    }

    expected_edges: set[frozenset[int]] = set()
    for a in range(lo, hi + 1):
        for d in range(1, hi - lo + 1):
            if a + 2 * d <= hi:
                expected_edges.add(frozenset({a, a + d, a + 2 * d}))

    assert edge_sets == expected_edges


# ---------------------------------------------------------------------------
# Reflection: [L,U] and [-U,-L] should have the same edge structure
# ---------------------------------------------------------------------------


def test_reflection_edge_count() -> None:
    r1 = construct_arithmetic_progression_hypergraph(1, 10, 3)
    r2 = construct_arithmetic_progression_hypergraph(-10, -1, 3)
    assert len(r1.hypergraph.edges) == len(r2.hypergraph.edges)
    assert len(r1.hypergraph.vertices) == len(r2.hypergraph.vertices)


# ---------------------------------------------------------------------------
# Large source: N=100, k=3
# ---------------------------------------------------------------------------


def test_large_source_n100_k3() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 100, 3)
    expected = _edge_count(100, 3)
    assert len(result.hypergraph.edges) == expected
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 3
    total = sum(len(m) for _, m in result.hypergraph.edges)
    assert total == 3 * expected


def test_large_source_n50_k4() -> None:
    result = construct_arithmetic_progression_hypergraph(0, 49, 4)
    expected = _edge_count(50, 4)
    assert len(result.hypergraph.edges) == expected
    for _eid, members in result.hypergraph.edges:
        assert len(members) == 4


# ---------------------------------------------------------------------------
# Result model fields
# ---------------------------------------------------------------------------


def test_result_preserves_request_fields() -> None:
    lo, hi, k = 3, 10, 3
    result = construct_arithmetic_progression_hypergraph(lo, hi, k)
    assert result.lower == lo
    assert result.upper == hi
    assert result.k == k


def test_result_k_field() -> None:
    result = construct_arithmetic_progression_hypergraph(1, 5, 4)
    assert result.k == 4


def test_native_constructor_rejects_oversized_vertex_carrier() -> None:
    with pytest.raises(OperationDomainValidationError, match="interval size"):
        construct_arithmetic_progression_hypergraph(
            0, MAX_INTERVAL_SIZE, MAX_INTERVAL_SIZE + 2
        )
