from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph
from jacobian.math.probability.hypergraph_edge_count_moments import (
    HypergraphEdgeCountMomentsResult,
    compute_hypergraph_edge_count_moments,
)


def run(vertices: tuple[str, ...], edges: tuple[tuple[str, tuple[str, ...]], ...], p: Fraction) -> HypergraphEdgeCountMomentsResult:
    return compute_hypergraph_edge_count_moments(FiniteHypergraph(vertices=vertices, edges=edges), CanonicalRational.from_fraction(p))

def oracle(vertices: tuple[str, ...], edges: tuple[tuple[str, tuple[str, ...]], ...], p: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    first = second = Fraction(0)
    for mask in range(1 << len(vertices)):
        retained = {v for i, v in enumerate(vertices) if mask >> i & 1}
        z = sum(members_set <= retained for _, members in edges for members_set in (set(members),))
        probability = p ** len(retained) * (1 - p) ** (len(vertices) - len(retained))
        first += probability * z
        second += probability * z * z
    return first, second, second - first * first

def test_exhaustive_subset_oracle_and_overlap_counts() -> None:
    vertices = ("a", "b", "c", "d")
    edges = (("e1", ("a", "b")), ("e2", ("b", "c")), ("e3", ("a", "b")), ("e4", ("d",)))
    for p in (Fraction(0), Fraction(1), Fraction(1, 2), Fraction(2, 3)):
        result = run(vertices, edges, p)
        expected = oracle(vertices, edges, p)
        assert tuple(value.as_fraction() for value in (result.edge_count_expectation, result.edge_count_second_moment, result.edge_count_variance)) == expected
    rows = {(row.edge_size_min, row.edge_size_max, row.intersection_size): row.pair_count for row in run(vertices, edges, Fraction(1, 2)).overlap_profile}
    assert rows[(2, 2, 2)] == 1  # duplicate host pair
    assert rows[(1, 2, 0)] == 3

def test_empty_edges_and_disjoint_nonuniform_profile() -> None:
    result = run(("a", "b", "c"), (("empty", ()), ("e1", ("a",)), ("e2", ("b", "c"))), Fraction(1, 2))
    assert result.edge_count_expectation.as_fraction() == Fraction(7, 4)
    assert result.edge_count_variance.as_fraction() == Fraction(7, 16)
    assert result.overlap_profile[0].intersection_size == 0
    assert result.overlap_profile[0].covariance.as_fraction() == 0

def test_accepted_large_compact_profile() -> None:
    vertices = tuple(f"v{i}" for i in range(101))
    edges = tuple((f"e{i}", (f"v{i}",)) for i in range(101))
    result = run(vertices, edges, Fraction(1, 2))
    assert len(result.overlap_profile) == 1
    assert result.overlap_profile[0].pair_count == 5050
    assert len(result.model_dump_json()) > 0
