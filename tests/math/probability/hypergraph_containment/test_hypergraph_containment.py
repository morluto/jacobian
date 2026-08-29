from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.probability.hypergraph_containment.operations import (
    compute_hypergraph_vertex_containment,
)


def _hg(vertices, edges):
    return FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple((eid, tuple(m)) for eid, m in edges),
    )


def test_single_edge_p1() -> None:
    """With p=1, every vertex is retained, probability 1."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1))
    )
    assert result.success_count == 1
    assert result.total_state_count == 4
    assert result.probability.as_fraction() == Fraction(1)


def test_single_edge_p0() -> None:
    """With p=0, no vertices retained, probability 0."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(0))
    )
    assert result.probability.as_fraction() == Fraction(0)


def test_single_edge_p_half() -> None:
    """With p=1/2: P(contains edge) = P(both retained) = (1/2)^2 = 1/4."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )
    assert result.probability.as_fraction() == Fraction(1, 4)
    assert result.success_count == 1


def test_empty_hypergraph() -> None:
    """Empty hypergraph: no edge can be contained, probability 0."""
    hg = _hg(["a", "b"], [])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1))
    )
    assert result.success_count == 0
    assert result.probability.as_fraction() == Fraction(0)


def test_two_edges() -> None:
    """Two edges: success = contains edge 0 OR contains edge 1."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b")), ("e1", ("b", "c"))])
    p = CanonicalRational.from_fraction(Fraction(1))
    result = compute_hypergraph_vertex_containment(hg, p)
    assert result.success_count == 3


def test_subset_counts_sum() -> None:
    """Sum of counts equals success_count."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b", "c"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 3))
    )
    assert sum(result.containing_subset_counts) == result.success_count


def test_total_state_count() -> None:
    """Total state count = 2^n."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )
    assert result.total_state_count == 8


def test_result_preserves_source() -> None:
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    p = CanonicalRational.from_fraction(Fraction(1, 2))
    result = compute_hypergraph_vertex_containment(hg, p)
    assert result.hypergraph == hg
    assert result.retention_probability == p
