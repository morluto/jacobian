from __future__ import annotations

from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph.operations import (
    compute_binary_union_hypergraph,
)


def test_simple_union() -> None:
    """{1} ∪ {2} = {1, 2}."""
    result = compute_binary_union_hypergraph(((1,), (2,), (1, 2)))
    # i=0, j=1, k=2: {1} ∪ {2} = {1,2} -> one relation
    assert result.relation_count == 1


def test_no_relations() -> None:
    """Disjoint sets with no union relation."""
    result = compute_binary_union_hypergraph(((1,), (2,), (3,)))
    assert result.relation_count == 0


def test_multiple_relations() -> None:
    """{1} ∪ {2} = {1,2}, {1} ∪ {3} = {1,3}, etc."""
    result = compute_binary_union_hypergraph(((1,), (2,), (3,), (1, 2), (1, 3), (2, 3)))
    # {1}∪{2}={1,2}, {1}∪{3}={1,3}, {2}∪{3}={2,3}
    assert result.relation_count == 3


def test_result_preserves_source() -> None:
    result = compute_binary_union_hypergraph(((1,), (2,), (1, 2)))
    assert result.sets == ((1,), (2,), (1, 2))
