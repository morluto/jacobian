"""Contract tests for orbit profiles of materialized subset families."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sympy.combinatorics import Permutation, PermutationGroup

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.actions._models import (
    ActionBoundSubset,
    FinitePermutationAction,
    SubsetFamilyOrbitProfileRequest,
    SubsetFamilyOrbitProfileResult,
)
from jacobian.math.groups.actions._tools import TOOLS
from jacobian.math.groups.actions.operations import (
    _enumerate_group,
    subset_family_orbit_profile,
    verify_subset_family_orbit_profile,
)


def _cyclic_c3() -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=("a", "b", "c"),
        generators=((1, 2, 0),),
    )


def _dihedral_d4() -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=("v0", "v1", "v2", "v3"),
        generators=((1, 2, 3, 0), (1, 0, 3, 2)),
    )


def _symmetric_s3() -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=("p0", "p1", "p2"),
        generators=((1, 2, 0), (1, 0, 2)),
    )


def _bound(
    action: FinitePermutationAction, positions: tuple[int, ...]
) -> ActionBoundSubset:
    return ActionBoundSubset(action=action, positions=positions)


def _request(
    action: FinitePermutationAction,
    *subsets: tuple[int, ...],
) -> SubsetFamilyOrbitProfileRequest:
    return SubsetFamilyOrbitProfileRequest(action=action, subsets=subsets)


def _result(
    action: FinitePermutationAction,
    *subsets: tuple[int, ...],
) -> SubsetFamilyOrbitProfileResult:
    return subset_family_orbit_profile(action, subsets)


def _assert_error_type(error: ValidationError, expected: str) -> None:
    assert error.errors()[0]["type"] == expected


def _transport(
    permutation: tuple[int, ...], subset: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(sorted(permutation[position] for position in subset))


def test_empty_family_returns_an_exact_empty_profile() -> None:
    result = _result(_cyclic_c3())

    assert result.family_size == 0
    assert result.rows == ()
    assert result.total_supplied_subsets == result.total_full_orbit_size == 0
    assert result.group_order == 3
    assert result.is_union_of_complete_orbits is True


def test_complete_singleton_family_reports_one_orbit_with_all_source_indices() -> None:
    action = _cyclic_c3()
    result = _result(action, (2,), (0,), (1,))

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.representative.positions == (0,)
    assert row.source_indices == (0, 1, 2)
    assert row.supplied_count == row.orbit_size == 3
    assert row.stabilizer_size == 1
    assert result.group_order == 3
    assert result.is_union_of_complete_orbits is True


def test_noninvariant_family_distinguishes_supplied_and_ambient_orbit_sizes() -> None:
    action = _dihedral_d4()
    result = _result(action, (0, 1), (2, 3))

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.representative.positions == (0, 1)
    assert row.source_indices == (0, 1)
    assert row.supplied_count == 2
    assert row.orbit_size == 4
    assert row.stabilizer_size == 2
    assert result.total_supplied_subsets == 2
    assert result.total_full_orbit_size == 4
    assert result.is_union_of_complete_orbits is False


def test_two_distinct_orbit_classes_partition_source_rows() -> None:
    action = _dihedral_d4()
    result = _result(action, (), (0, 2), (0, 1, 2, 3), (0,))

    assert tuple(row.representative.positions for row in result.rows) == (
        (),
        (0,),
        (0, 1, 2, 3),
        (0, 2),
    )
    assert [row.source_indices for row in result.rows] == [(0,), (3,), (2,), (1,)]
    assert [row.orbit_size for row in result.rows] == [1, 4, 1, 2]
    assert [row.stabilizer_size for row in result.rows] == [8, 2, 8, 4]
    assert result.is_union_of_complete_orbits is False
    assert result.total_full_orbit_size == 8


def test_nonfaithful_action_has_nontrivial_point_stabilizers() -> None:
    action = _symmetric_s3()
    action_with_fixed_point = FinitePermutationAction(
        domain=action.domain,
        generators=(action.generators[0],),
    )
    result = _result(action_with_fixed_point, (0,), (1,))

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.representative.positions == (0,)
    assert row.source_indices == (0, 1)
    assert row.supplied_count == 2
    assert row.orbit_size == 3
    assert row.stabilizer_size == 1
    assert result.group_order == 3
    assert result.is_union_of_complete_orbits is False


def test_empty_and_full_subsets_have_whole_group_stabilizers() -> None:
    action = _dihedral_d4()
    result = _result(action, (), (0, 1, 2, 3))

    assert [row.orbit_size for row in result.rows] == [1, 1]
    assert [row.stabilizer_size for row in result.rows] == [8, 8]
    assert result.is_union_of_complete_orbits is True


@pytest.mark.property
@pytest.mark.parametrize(
    "action",
    [_cyclic_c3(), _symmetric_s3(), _dihedral_d4()],
)
def test_profile_agrees_with_sympy_orbit_partition(
    action: FinitePermutationAction,
) -> None:
    """Compare against independent SymPy enumeration for every singleton subset."""

    result = _result(action, *((position,) for position in range(len(action.domain))))
    elements = tuple(
        tuple(permutation)
        for permutation in PermutationGroup(
            [Permutation(list(generator)) for generator in action.generators]
        ).generate_schreier_sims(af=True)
    )
    expected: dict[tuple[int, ...], set[int]] = {}
    for index, position in enumerate(range(len(action.domain))):
        subset = (position,)
        orbit = {_transport(permutation, subset) for permutation in elements}
        expected.setdefault(min(orbit), set()).add(index)

    assert tuple(row.representative.positions for row in result.rows) == tuple(
        sorted(expected)
    )
    assert all(
        tuple(sorted(expected[row.representative.positions])) == row.source_indices
        for row in result.rows
    )
    assert all(
        row.orbit_size * row.stabilizer_size == len(_enumerate_group(action))
        for row in result.rows
    )


def test_generator_reordering_preserves_the_mathematical_profile() -> None:
    first = _symmetric_s3()
    second = FinitePermutationAction(
        domain=first.domain,
        generators=tuple(reversed(first.generators)),
    )
    first_result = _result(first, (0,), (0, 1))
    second_result = _result(second, (0,), (0, 1))

    assert first_result.group_order == second_result.group_order
    assert first_result.family_size == second_result.family_size
    assert first_result.is_union_of_complete_orbits == (
        second_result.is_union_of_complete_orbits
    )
    assert first_result.total_supplied_subsets == second_result.total_supplied_subsets
    assert first_result.total_full_orbit_size == second_result.total_full_orbit_size
    assert tuple(
        (
            row.representative.positions,
            row.source_indices,
            row.supplied_count,
            row.orbit_size,
            row.stabilizer_size,
        )
        for row in first_result.rows
    ) == tuple(
        (
            row.representative.positions,
            row.source_indices,
            row.supplied_count,
            row.orbit_size,
            row.stabilizer_size,
        )
        for row in second_result.rows
    )


def test_domain_relabelling_changes_only_the_bound_action_representation() -> None:
    first = _symmetric_s3()
    relabelled = FinitePermutationAction(
        domain=("x", "y", "z"),
        generators=first.generators,
    )
    first_result = _result(first, (0,))
    relabelled_result = _result(relabelled, (0,))

    assert first_result.group_order == relabelled_result.group_order
    assert first_result.family_size == relabelled_result.family_size
    assert (
        first_result.is_union_of_complete_orbits
        == relabelled_result.is_union_of_complete_orbits
    )
    assert (
        first_result.rows[0].representative.positions
        == relabelled_result.rows[0].representative.positions
    )
    assert (
        first_result.rows[0].source_indices == relabelled_result.rows[0].source_indices
    )


def test_request_rejects_invalid_and_out_of_domain_positions() -> None:
    action = _cyclic_c3()
    with pytest.raises(ValidationError) as exc_info:
        _request(action, (0, 0))
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_positions_not_distinct"
    )
    with pytest.raises(ValidationError) as exc_info:
        _request(action, (3,))
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_position_out_of_range"
    )


def test_request_rejects_duplicate_family_members() -> None:
    action = _cyclic_c3()
    with pytest.raises(ValidationError) as exc_info:
        _request(action, (0, 1), (0, 1))
    _assert_error_type(exc_info.value, "finite_group_action.subset_family_duplicates")

    with pytest.raises(ValidationError) as exc_info:
        _request(action, (0,), (0,))
    _assert_error_type(exc_info.value, "finite_group_action.subset_family_duplicates")


def test_request_rejects_group_immediately_above_enumeration_bound() -> None:
    # S_9 has order 362880 and S_10 has order 3628800, so this crosses the
    # existing 10000-element action bound without relying on factorial text.
    degree = 9
    action = FinitePermutationAction(
        domain=tuple(f"p{position}" for position in range(degree)),
        generators=((*range(1, degree), 0), (1, 0, *range(2, degree))),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        _result(action, (0,))

    assert error.value.errors()[0]["type"] == (
        "finite_group_action.group_order_exceeds_bound"
    )
    assert error.value.errors()[0]["loc"] == ("action",)


@pytest.mark.parametrize(
    ("forged_field", "value"),
    [
        ("source_indices", (0, 0)),
        ("supplied_count", 2),
        ("orbit_size", 2),
        ("stabilizer_size", 2),
    ],
)
def test_result_rejects_inconsistent_row_claims(
    forged_field: str, value: tuple[int, ...] | int
) -> None:
    result = _result(_cyclic_c3(), (0,))
    forged = result.model_dump()
    if isinstance(value, tuple):
        forged["rows"][0][forged_field] = list(value)
    else:
        forged["rows"][0][forged_field] = value

    if forged_field == "source_indices":
        with pytest.raises(ValidationError):
            SubsetFamilyOrbitProfileResult.model_validate(forged)
    else:
        claim = SubsetFamilyOrbitProfileResult.model_validate(forged)
        assert not verify_subset_family_orbit_profile(claim)


def test_result_rejects_lost_source_coverage_and_wrong_completeness() -> None:
    action = _cyclic_c3()
    result = _result(action, (0,), (1,))
    forged = result.model_dump()
    forged["rows"][0]["source_indices"] = [0]
    forged["rows"][0]["supplied_count"] = 1
    forged["is_union_of_complete_orbits"] = True
    forged["total_supplied_subsets"] = 1

    claim = SubsetFamilyOrbitProfileResult.model_validate(forged)
    assert not verify_subset_family_orbit_profile(claim)


def test_result_rejects_duplicate_orbit_representatives() -> None:
    result = _result(_dihedral_d4(), (), (0, 1, 2, 3))
    forged = result.model_dump()
    forged["rows"][0]["representative"] = forged["rows"][1]["representative"]

    claim = SubsetFamilyOrbitProfileResult.model_validate(forged)
    assert not verify_subset_family_orbit_profile(claim)


def test_result_rejects_representative_bound_to_a_different_action() -> None:
    result = _result(_cyclic_c3(), (0,))
    forged = result.model_dump()
    forged["rows"][0]["representative"]["action"] = _dihedral_d4().model_dump()

    claim = SubsetFamilyOrbitProfileResult.model_validate(forged)
    assert not verify_subset_family_orbit_profile(claim)


def test_serialized_result_round_trips_unchanged() -> None:
    action = _dihedral_d4()
    result = _result(action, (0, 1), (2, 3))

    restored = SubsetFamilyOrbitProfileResult.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result
    assert verify_subset_family_orbit_profile(restored)


def test_public_declaration_exposes_and_executes_copyable_example() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "group_action.subset_family.orbit_profile.compute"
    )

    assert operation.examples
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)

    assert isinstance(result, SubsetFamilyOrbitProfileResult)
    assert result.family_size == 1
    assert result.rows[0].representative.positions == (0,)
    assert result.rows[0].orbit_size == 3
    schema = operation.request_type.model_json_schema()
    assert schema["properties"]["subsets"]["maxItems"] == 5000
    assert "supplied action domain" in schema["properties"]["subsets"]["description"]
    assert set(operation.examples[0].input) == {"action", "subsets"}
