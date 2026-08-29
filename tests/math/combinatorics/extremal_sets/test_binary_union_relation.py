from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationRequest,
)
from jacobian.math.combinatorics.extremal_sets.operations import (
    construct_binary_union_relation,
)


def test_fixture_boolean_lattice() -> None:
    """Fixture: {empty, {a}, {b}, {a,b}} has one union: {a} U {b} = {a,b}."""
    family = ((), (0,), (1,), (0, 1))
    result = construct_binary_union_relation(family)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.operand_i == 1
    assert row.operand_j == 2
    assert row.result_k == 3


def test_empty_family_rejected() -> None:
    with pytest.raises(ValidationError):
        BinaryUnionRelationRequest(family=())


def test_single_member() -> None:
    """A singleton family has no union relations."""
    result = construct_binary_union_relation(((),))
    assert len(result.rows) == 0


def test_two_members_no_relation() -> None:
    """Two distinct sets with no third member: no union relation."""
    result = construct_binary_union_relation(((0,), (1,)))
    assert len(result.rows) == 0


def test_positive_relation() -> None:
    """{0} U {1} = {0,1}."""
    family = ((0,), (1,), (0, 1))
    result = construct_binary_union_relation(family)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.operand_i == 0
    assert row.operand_j == 1
    assert row.result_k == 2


def test_no_relation_chains() -> None:
    """Family with no union triples."""
    family = ((0,), (1,), (2,))
    result = construct_binary_union_relation(family)
    assert len(result.rows) == 0


def test_rejects_duplicate_members() -> None:
    with pytest.raises(ValidationError):
        BinaryUnionRelationRequest(family=((0,), (0,)))


def test_rejects_unsorted_elements() -> None:
    with pytest.raises(ValidationError):
        BinaryUnionRelationRequest(family=((1, 0),))


def test_rejects_duplicate_elements() -> None:
    with pytest.raises(ValidationError):
        BinaryUnionRelationRequest(family=((0, 0),))


def test_row_replay() -> None:
    """Replay: S_i union S_j = S_k for every row."""
    family = ((), (0,), (1,), (0, 1), (0, 1, 2))
    result = construct_binary_union_relation(family)
    sets = [frozenset(s) for s in family]
    for row in result.rows:
        assert sets[row.operand_i] | sets[row.operand_j] == sets[row.result_k]
        assert row.operand_i < row.operand_j
        assert row.result_k != row.operand_i
        assert row.result_k != row.operand_j


def test_exhaustive_comparison() -> None:
    """Compare against independent nested pair loop."""
    family = ((), (0,), (1,), (0, 1))
    result = construct_binary_union_relation(family)
    sets = [frozenset(s) for s in family]
    expected = []
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            union = sets[i] | sets[j]
            for k in range(len(family)):
                if k != i and k != j and sets[k] == union:
                    expected.append((i, j, k))
    assert len(result.rows) == len(expected)
    for row, exp in zip(result.rows, expected, strict=True):
        assert (row.operand_i, row.operand_j, row.result_k) == exp


def test_hypergraph_edges() -> None:
    """The hypergraph has one 3-edge per union row."""
    family = ((), (0,), (1,), (0, 1))
    result = construct_binary_union_relation(family)
    assert len(result.hypergraph.edges) == 1
    _, members = next(iter(result.hypergraph.edges))
    assert set(members) == {"1", "2", "3"}


def test_source_preserved() -> None:
    """Result retains the original family."""
    family = ((0,), (1,))
    result = construct_binary_union_relation(family)
    assert result.family == family
