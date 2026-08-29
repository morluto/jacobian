from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.nonlinear._models import ExplicitBinaryCode
from jacobian.math.combinatorics.codes.nonlinear.operations import to_set_system
from jacobian.math.combinatorics.extremal_sets.operations import (
    construct_binary_union_relation,
)
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily


def _source(
    members: tuple[tuple[int, ...], ...], ground_set_size: int = 3
) -> IndexedFiniteSetFamily:
    return IndexedFiniteSetFamily(
        ground_set_size=ground_set_size,
        members=members,
    )


def test_boolean_lattice_fixture_preserves_orientation() -> None:
    source = _source(((), (0,), (1,), (0, 1)), ground_set_size=2)
    result = construct_binary_union_relation(source)
    assert result.source == source
    assert result.rows[0].model_dump() == {
        "edge_id": "union_1_2_to_3",
        "operand_i": 1,
        "operand_j": 2,
        "result_k": 3,
    }
    assert result.hypergraph.edges == (("union_1_2_to_3", ("1", "2", "3")),)


def test_empty_family_is_an_exact_empty_relation() -> None:
    source = _source((), ground_set_size=2)
    result = construct_binary_union_relation(source)
    assert result.source == source
    assert result.rows == ()
    assert result.hypergraph.vertices == ()
    assert result.hypergraph.edges == ()


def test_unused_ground_axis_is_retained() -> None:
    source = _source(((0,), (1,)), ground_set_size=3)
    result = construct_binary_union_relation(source)
    assert result.source.ground_set_size == 3
    assert result.rows == ()


@pytest.mark.parametrize(
    "members",
    [
        ((0,), (0,)),
        ((1, 0),),
        ((0, 0),),
        ((3,),),
    ],
)
def test_source_rejects_noncanonical_members(
    members: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(ValidationError):
        _source(members)


def test_every_row_replays_and_the_relation_is_complete() -> None:
    source = _source(((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)))
    result = construct_binary_union_relation(source)
    sets = tuple(frozenset(member) for member in source.members)
    expected = tuple(
        (i, j, k)
        for i, j in combinations(range(len(sets)), 2)
        for k in range(len(sets))
        if k not in (i, j) and sets[i] | sets[j] == sets[k]
    )
    actual = tuple((row.operand_i, row.operand_j, row.result_k) for row in result.rows)
    assert actual == expected
    assert len({row.edge_id for row in result.rows}) == len(result.rows)
    assert {edge_id for edge_id, _ in result.hypergraph.edges} == {
        row.edge_id for row in result.rows
    }


def test_input_permutation_covariance() -> None:
    first = _source(((0,), (1,), (0, 1)), ground_set_size=2)
    second = _source(((0, 1), (1,), (0,)), ground_set_size=2)
    first_result = construct_binary_union_relation(first)
    second_result = construct_binary_union_relation(second)
    assert first_result.rows[0].model_dump(exclude={"edge_id"}) == {
        "operand_i": 0,
        "operand_j": 1,
        "result_k": 2,
    }
    assert second_result.rows[0].model_dump(exclude={"edge_id"}) == {
        "operand_i": 1,
        "operand_j": 2,
        "result_k": 0,
    }


def test_dense_relation_exceeding_hypergraph_carrier_rejects_before_result() -> None:
    members = tuple(
        combination
        for size in range(9)
        if size != 3
        for combination in combinations(range(8), size)
    )
    source = IndexedFiniteSetFamily(ground_set_size=8, members=members)
    assert len(source.members) == 200
    with pytest.raises(OperationDomainValidationError) as error:
        construct_binary_union_relation(source)
    assert error.value.errors()[0]["type"] == (
        "set_system.binary_union_relation.result_exceeds_carrier"
    )


def test_large_sparse_family_remains_admitted() -> None:
    source = IndexedFiniteSetFamily(
        ground_set_size=200,
        members=tuple((index,) for index in range(200)),
    )
    result = construct_binary_union_relation(source)
    assert result.rows == ()
    assert len(result.hypergraph.vertices) == 200


def test_code_support_producer_composes_without_reconstruction() -> None:
    code = ExplicitBinaryCode(
        length=2,
        codewords=((0, 0), (1, 0), (0, 1), (1, 1)),
    )
    result = construct_binary_union_relation(to_set_system(code))
    assert result.source.ground_set_size == 2
    assert result.rows[0].model_dump(exclude={"edge_id"}) == {
        "operand_i": 1,
        "operand_j": 2,
        "result_k": 3,
    }


def test_ground_axis_is_independent_of_relation_vertex_count() -> None:
    result = construct_binary_union_relation(
        _source(((256,),), ground_set_size=257)
    )
    assert result.source.ground_set_size == 257
    assert result.rows == ()
