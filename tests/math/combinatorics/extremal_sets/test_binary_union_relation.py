from __future__ import annotations

import json
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.nonlinear.operations import to_set_system
from jacobian.math.combinatorics.codes.nonlinear.values import ExplicitBinaryCode
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationRequest,
)
from jacobian.math.combinatorics.extremal_sets.operations import (
    MAX_BINARY_UNION_MEMBERSHIP_WORK,
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

    serialized_request = BinaryUnionRelationRequest.model_validate(
        {"source": to_set_system(code).model_dump(mode="json")}
    )
    assert (
        construct_binary_union_relation(serialized_request.source).rows == result.rows
    )


def test_oversized_code_support_family_is_a_domain_rejection() -> None:
    length = 512
    code = ExplicitBinaryCode(
        length=length,
        codewords=tuple(
            tuple(1 if coordinate == word else 0 for coordinate in range(length))
            for word in range(length)
        ),
    )

    with pytest.raises(OperationDomainValidationError, match="relation carrier"):
        construct_binary_union_relation(to_set_system(code))


def test_noncanonical_code_support_is_a_domain_rejection() -> None:
    code = ExplicitBinaryCode(length=2, codewords=((1, 1),))
    payload = to_set_system(code).model_dump(mode="json")
    payload["members"] = [[1, 0]]
    with pytest.raises(ValidationError, match="strictly increasing"):
        IndexedFiniteSetFamily.model_validate_json(json.dumps(payload))


def test_support_target_is_an_ordinary_set_family() -> None:
    code = ExplicitBinaryCode(
        length=2,
        codewords=((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    payload = to_set_system(code).model_dump(mode="json")
    target = IndexedFiniteSetFamily.model_validate_json(json.dumps(payload))
    assert construct_binary_union_relation(target).source == target


def test_serialized_code_support_axis_is_bound_to_source_coordinates() -> None:
    code = ExplicitBinaryCode(length=2, codewords=((0, 0), (1, 0)))
    payload = to_set_system(code).model_dump(mode="json")
    payload["ground_set_size"] = 0

    with pytest.raises(ValidationError, match="ground_set_size"):
        type(to_set_system(code)).model_validate_json(json.dumps(payload))


def test_total_membership_work_is_bounded_before_pair_scanning() -> None:
    common_size = MAX_BINARY_UNION_MEMBERSHIP_WORK // (255 * 256) + 1
    common = tuple(range(common_size))
    source = IndexedFiniteSetFamily(
        ground_set_size=common_size + 256,
        members=tuple((*common, common_size + index) for index in range(256)),
    )

    with pytest.raises(OperationDomainValidationError, match="membership work"):
        construct_binary_union_relation(source)


def test_generated_union_hashing_is_charged_to_membership_work() -> None:
    source = _source(
        tuple(
            tuple(index * 200 + offset for offset in range(200)) for index in range(256)
        ),
        ground_set_size=51_200,
    )

    with pytest.raises(OperationDomainValidationError, match="membership work"):
        construct_binary_union_relation(source)


def test_single_member_normalization_is_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source((tuple(range(32)),), ground_set_size=32)
    monkeypatch.setattr(
        "jacobian.math.combinatorics.extremal_sets.operations.MAX_BINARY_UNION_MEMBERSHIP_WORK",
        31,
    )

    with pytest.raises(OperationDomainValidationError, match="membership work"):
        construct_binary_union_relation(source)


def test_ground_axis_is_independent_of_relation_vertex_count() -> None:
    result = construct_binary_union_relation(_source(((256,),), ground_set_size=257))
    assert result.source.ground_set_size == 257
    assert result.rows == ()


@pytest.mark.parametrize(
    "payload",
    [
        {"ground_set_size": True, "members": [[0]]},
        {"ground_set_size": 1, "members": [[False]]},
    ],
)
def test_canonical_family_rejects_boolean_coordinates(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        IndexedFiniteSetFamily.model_validate_json(json.dumps(payload))
