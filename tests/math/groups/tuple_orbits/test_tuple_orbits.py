"""Diagonal tuple-family orbit profiles."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.actions._models import (
    MAX_FAMILY_MEMBERS,
    FinitePermutationAction,
)
from jacobian.math.groups.tuple_orbits._models import (
    MAX_TUPLE_ARITY,
    TupleFamilyOrbitResult,
    TupleFamilyOrbitSource,
    TupleOrbitRow,
)
from jacobian.math.groups.tuple_orbits.operations import tuple_family_orbit_profile


def _swap_action() -> FinitePermutationAction:
    return FinitePermutationAction(domain=("a", "b"), generators=((1, 0),))


def _cyclic_action() -> FinitePermutationAction:
    return FinitePermutationAction(domain=("a", "b", "c"), generators=((1, 2, 0),))


def test_repeated_coordinates_and_duplicate_sources_are_retained() -> None:
    request = TupleFamilyOrbitSource(
        action=_cyclic_action(),
        arity=2,
        family=((2, 2), (0, 0), (0, 1), (1, 2), (0, 1)),
    )
    result = tuple_family_orbit_profile(request)
    assert [row.representative for row in result.rows] == [(0, 0), (0, 1)]
    assert result.rows[0].source_indices == (0, 1)
    assert result.rows[1].source_indices == (2, 3, 4)
    assert all(row.orbit_size == 3 and row.stabilizer_size == 1 for row in result.rows)
    assert result.is_union_of_complete_ambient_orbits is False


def test_complete_ambient_orbits_are_reported() -> None:
    request = TupleFamilyOrbitSource(
        action=_cyclic_action(),
        arity=2,
        family=((0, 0), (1, 1), (2, 2)),
    )
    result = tuple_family_orbit_profile(request)
    assert result.is_union_of_complete_ambient_orbits is True


def test_empty_family_has_empty_profile() -> None:
    request = TupleFamilyOrbitSource(
        action=FinitePermutationAction(domain=("a",), generators=((0,),)),
        arity=0,
        family=(),
    )
    assert tuple_family_orbit_profile(request).rows == ()


def test_empty_family_skips_generated_group_admission() -> None:
    request = TupleFamilyOrbitSource(
        action=FinitePermutationAction(
            domain=tuple(str(index) for index in range(8)),
            generators=(
                (1, 2, 3, 4, 5, 6, 7, 0),
                (1, 0, 2, 3, 4, 5, 6, 7),
            ),
        ),
        arity=0,
        family=(),
    )
    result = tuple_family_orbit_profile(request)
    assert result.rows == ()
    assert result.is_union_of_complete_ambient_orbits is True


def test_action_bound_order_and_repeated_coordinates_are_exact() -> None:
    request = TupleFamilyOrbitSource(
        action=_swap_action(),
        arity=2,
        family=((0, 1), (1, 0), (0, 0), (1, 1)),
    )
    result = tuple_family_orbit_profile(request)
    assert [row.representative for row in result.rows] == [(0, 0), (0, 1)]
    assert result.rows[0].source_indices == (2, 3)
    assert result.rows[1].source_indices == (0, 1)
    assert all(row.orbit_size == 2 and row.stabilizer_size == 1 for row in result.rows)
    assert result.rows[0].least_transporter == (0, 1)
    assert result.rows[1].least_transporter == (0, 1)
    assert result.is_union_of_complete_ambient_orbits


def test_tuple_coordinate_order_remains_distinct_for_the_trivial_action() -> None:
    request = TupleFamilyOrbitSource(
        action=FinitePermutationAction(domain=("a", "b"), generators=((0, 1),)),
        arity=2,
        family=((0, 1), (1, 0), (0, 0)),
    )
    result = tuple_family_orbit_profile(request)
    assert [row.representative for row in result.rows] == [(0, 0), (0, 1), (1, 0)]
    assert all(row.orbit_size == row.stabilizer_size == 1 for row in result.rows)


def test_duplicate_source_rows_do_not_fake_complete_orbit_coverage() -> None:
    request = TupleFamilyOrbitSource(
        action=_swap_action(),
        arity=2,
        family=((0, 0), (1, 1), (0, 0)),
    )
    result = tuple_family_orbit_profile(request)
    assert result.rows[0].source_indices == (0, 1, 2)
    assert not result.is_union_of_complete_ambient_orbits


def test_serialized_result_preserves_axes_and_transporter_replay() -> None:
    result = tuple_family_orbit_profile(
        TupleFamilyOrbitSource(action=_swap_action(), arity=2, family=((1, 0), (0, 1)))
    )
    restored = type(result).model_validate(result.model_dump())
    assert restored == result
    assert (
        type(result).model_validate_json(result.model_dump_json(), strict=True)
        == result
    )
    row = result.rows[0]
    assert (
        tuple(
            row.least_transporter[position]
            for position in result.source.family[row.source_indices[0]]
        )
        == row.representative
    )

    forged = result.model_dump()
    forged["rows"][0]["least_transporter"] = [0, 1]
    with pytest.raises(ValidationError):
        type(result).model_validate(forged)


def test_large_generated_group_is_rejected_before_element_materialization() -> None:
    request = TupleFamilyOrbitSource(
        action=FinitePermutationAction(
            domain=tuple(str(index) for index in range(8)),
            generators=(
                (1, 2, 3, 4, 5, 6, 7, 0),
                (1, 0, 2, 3, 4, 5, 6, 7),
            ),
        ),
        arity=2,
        family=((0, 0),),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        tuple_family_orbit_profile(request)
    assert exc_info.value.errors()[0]["type"] == (
        "finite_group_action.tuple_family_group_order_bound"
    )


def test_forged_action_fields_are_revalidated_before_backend_conversion() -> None:
    forged_action = FinitePermutationAction.model_construct(
        domain=("a", "a"), generators=((0, 1),)
    )
    request = TupleFamilyOrbitSource.model_construct(
        action=forged_action, arity=1, family=((0,),)
    )
    with pytest.raises(OperationDomainValidationError) as labels:
        tuple_family_orbit_profile(request)
    assert "domain_labels_not_distinct" in labels.value.errors()[0]["type"]

    malformed = FinitePermutationAction.model_construct(
        domain=("a", "b"), generators=((0,),)
    )
    malformed_request = TupleFamilyOrbitSource.model_construct(
        action=malformed, arity=1, family=((0,),)
    )
    with pytest.raises(OperationDomainValidationError) as generators:
        tuple_family_orbit_profile(malformed_request)
    assert "generator" in generators.value.errors()[0]["type"]


def test_result_source_retains_the_revalidated_action() -> None:
    coerced_action = FinitePermutationAction.model_construct(
        domain=("a", "b"), generators=(("1", "0"),)
    )
    request = TupleFamilyOrbitSource.model_construct(
        action=coerced_action, arity=1, family=((0,), (1,))
    )
    result = tuple_family_orbit_profile(request)
    assert result.source.action.generators == ((1, 0),)
    assert result.source.action.domain == ("a", "b")


def test_raw_family_dimensions_are_rejected_before_container_copy() -> None:
    action = {"domain": ["a"], "generators": [[0]]}
    oversized = {
        "action": action,
        "arity": 0,
        "family": [[] for _ in range(MAX_FAMILY_MEMBERS + 1)],
    }
    with pytest.raises(ValidationError) as family_bound:
        TupleFamilyOrbitSource.model_validate(oversized)
    assert "input_bound" in family_bound.value.errors()[0]["type"]

    class _HugeRow(tuple):
        def __len__(self) -> int:
            return MAX_TUPLE_ARITY + 1

    oversized_row = {
        "action": action,
        "arity": 2,
        "family": [_HugeRow()],
    }
    with pytest.raises(ValidationError) as arity_mismatch:
        TupleFamilyOrbitSource.model_validate(oversized_row)
    assert "arity_mismatch" in arity_mismatch.value.errors()[0]["type"]


def test_range_generator_rows_are_rejected_before_container_copy() -> None:
    payload = {
        "action": {"domain": ["a"], "generators": [range(2_000_000)]},
        "arity": 0,
        "family": [],
    }
    with pytest.raises(ValidationError) as generator_length:
        TupleFamilyOrbitSource.model_validate(payload)
    assert "generator_length_mismatch" in generator_length.value.errors()[0]["type"]


def test_constructed_nested_action_is_revalidated() -> None:
    payload = {
        "action": FinitePermutationAction.model_construct(
            domain=("a", "a"), generators=((0, 1),)
        ),
        "arity": 0,
        "family": [],
    }
    with pytest.raises(ValidationError) as labels:
        TupleFamilyOrbitSource.model_validate(payload)
    assert "domain_labels_not_distinct" in labels.value.errors()[0]["type"]


def test_unknown_action_fields_are_preserved_for_strict_rejection() -> None:
    payload = {
        "action": {"domain": ["a"], "generators": [[0]], "generator": [[0]]},
        "arity": 0,
        "family": [],
    }
    with pytest.raises(ValidationError) as extra:
        TupleFamilyOrbitSource.model_validate(payload)
    assert extra.value.errors()[0]["type"] == "extra_forbidden"


def test_unknown_action_fields_on_result_source_are_rejected() -> None:
    payload = {
        "source": {
            "action": {"domain": ["a"], "generators": [[0]], "generator": [[0]]},
            "arity": 0,
            "family": [],
        },
        "rows": [],
        "is_union_of_complete_ambient_orbits": True,
    }
    with pytest.raises(ValidationError) as extra:
        TupleFamilyOrbitResult.model_validate(payload)
    assert extra.value.errors()[0]["type"] == "extra_forbidden"


def test_range_family_rows_are_rejected_before_container_copy() -> None:
    payload = {
        "action": {"domain": ["a"], "generators": [[0]]},
        "arity": 0,
        "family": [range(2_000_000)],
    }
    with pytest.raises(ValidationError) as arity:
        TupleFamilyOrbitSource.model_validate(payload)
    assert arity.value.errors()[0]["type"] in {
        "finite_group_action.tuple_family_arity_out_of_range",
        "finite_group_action.tuple_family_arity_mismatch",
    }


def test_raw_action_generator_dimensions_are_rejected_before_container_copy() -> None:
    class _HugeGenerator(tuple):
        def __len__(self) -> int:
            return 2_000_000

    payload = {
        "action": {"domain": ["a"], "generators": [_HugeGenerator()]},
        "arity": 0,
        "family": [],
    }
    with pytest.raises(ValidationError) as generator_length:
        TupleFamilyOrbitSource.model_validate(payload)
    assert "generator_length_mismatch" in generator_length.value.errors()[0]["type"]


def test_nested_constructed_action_is_preflighted_on_source_validate() -> None:
    class _HugeGenerator(tuple):
        def __len__(self) -> int:
            return 2_000_000

    payload = {
        "action": FinitePermutationAction.model_construct(
            domain=("a",), generators=(_HugeGenerator(),)
        ),
        "arity": 0,
        "family": [],
    }
    with pytest.raises(ValidationError) as generator_length:
        TupleFamilyOrbitSource.model_validate(payload)
    assert "generator_length_mismatch" in generator_length.value.errors()[0]["type"]


def test_forged_action_generators_are_preflighted_before_pydantic() -> None:
    class _HugeGenerator(tuple):
        def __len__(self) -> int:
            return 2_000_000

    forged_action = FinitePermutationAction.model_construct(
        domain=("a",), generators=(_HugeGenerator(),)
    )
    request = TupleFamilyOrbitSource.model_construct(
        action=forged_action, arity=0, family=()
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tuple_family_orbit_profile(request)
    assert "generator_length_mismatch" in exc_info.value.errors()[0]["type"]


def test_missing_action_on_forged_source_is_a_typed_domain_error() -> None:
    request = TupleFamilyOrbitSource.model_construct(arity=0, family=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tuple_family_orbit_profile(request)
    assert exc_info.value.errors()[0]["type"] == (
        "finite_group_action.tuple_family_action_type"
    )


def test_incomplete_forged_action_is_a_typed_domain_error() -> None:
    forged_action = FinitePermutationAction.model_construct(generators=((0,),))
    request = TupleFamilyOrbitSource.model_construct(
        action=forged_action, arity=0, family=()
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tuple_family_orbit_profile(request)
    assert exc_info.value.errors()[0]["type"] == (
        "finite_group_action.tuple_family_action_type"
    )


def test_orbit_row_payloads_are_preflighted_before_container_copy() -> None:
    class _HugeRepresentative(tuple):
        def __len__(self) -> int:
            return 2_000_000

    payload = {
        "representative": _HugeRepresentative(),
        "source_indices": [0],
        "orbit_size": 1,
        "stabilizer_size": 1,
        "least_transporter": [0],
    }
    with pytest.raises(ValidationError) as arity:
        TupleOrbitRow.model_validate(payload)
    assert "arity_out_of_range" in arity.value.errors()[0]["type"]


def test_result_payloads_are_preflighted_before_container_copy() -> None:
    payload = {
        "source": {
            "action": {"domain": ["a"], "generators": [[0]]},
            "arity": 0,
            "family": [],
        },
        "rows": [{} for _ in range(MAX_FAMILY_MEMBERS + 1)],
        "is_union_of_complete_ambient_orbits": True,
    }
    with pytest.raises(ValidationError) as rows_bound:
        TupleFamilyOrbitResult.model_validate(payload)
    assert "input_bound" in rows_bound.value.errors()[0]["type"]


def test_missing_arity_on_forged_source_is_a_typed_domain_error() -> None:
    request = TupleFamilyOrbitSource.model_construct(action=_swap_action(), family=())
    with pytest.raises(OperationDomainValidationError) as exc_info:
        tuple_family_orbit_profile(request)
    assert exc_info.value.errors()[0]["type"] == (
        "finite_group_action.tuple_family_arity_out_of_range"
    )


def test_constructed_result_rows_are_revalidated() -> None:
    class _HugeRepresentative(tuple):
        def __len__(self) -> int:
            return 2_000_000

    row = TupleOrbitRow.model_construct(
        representative=_HugeRepresentative(),
        source_indices=(0,),
        orbit_size=1,
        stabilizer_size=1,
        least_transporter=(0,),
    )
    payload = {
        "source": {
            "action": {"domain": ["a"], "generators": [[0]]},
            "arity": 0,
            "family": [],
        },
        "rows": [row],
        "is_union_of_complete_ambient_orbits": True,
    }
    with pytest.raises(ValidationError) as arity:
        TupleFamilyOrbitResult.model_validate(payload)
    assert "arity_out_of_range" in arity.value.errors()[0]["type"]


def test_constructed_result_rows_reject_noncanonical_orbit_size() -> None:
    row = TupleOrbitRow.model_construct(
        representative=(),
        source_indices=(0,),
        orbit_size="bogus",
        stabilizer_size=1,
        least_transporter=(0,),
    )
    payload = {
        "source": {
            "action": {"domain": ["a"], "generators": [[0]]},
            "arity": 0,
            "family": [()],
        },
        "rows": [row],
        "is_union_of_complete_ambient_orbits": True,
    }
    with pytest.raises(ValidationError):
        TupleFamilyOrbitResult.model_validate(payload)


def test_constructed_result_instance_is_revalidated() -> None:
    row = TupleOrbitRow.model_construct(
        representative=(),
        source_indices=(0,),
        orbit_size="bogus",
        stabilizer_size=1,
        least_transporter=(0,),
    )
    result = TupleFamilyOrbitResult.model_construct(
        source=TupleFamilyOrbitSource(
            action=FinitePermutationAction(domain=("a",), generators=((0,),)),
            arity=0,
            family=((),),
        ),
        rows=(row,),
        is_union_of_complete_ambient_orbits=True,
    )
    with pytest.raises(ValidationError):
        TupleFamilyOrbitResult.model_validate(result)


def test_aggregate_source_indices_are_bounded_before_container_copy() -> None:
    payload = {
        "source": {
            "action": {"domain": ["a"], "generators": [[0]]},
            "arity": 0,
            "family": [],
        },
        "rows": [
            {
                "representative": [0],
                "source_indices": list(range(MAX_FAMILY_MEMBERS)),
                "orbit_size": 1,
                "stabilizer_size": 1,
                "least_transporter": [0],
            },
            {
                "representative": [1],
                "source_indices": list(range(MAX_FAMILY_MEMBERS)),
                "orbit_size": 1,
                "stabilizer_size": 1,
                "least_transporter": [0],
            },
        ],
        "is_union_of_complete_ambient_orbits": True,
    }
    with pytest.raises(ValidationError) as indices_bound:
        TupleFamilyOrbitResult.model_validate(payload)
    assert "input_bound" in indices_bound.value.errors()[0]["type"]


def test_complete_orbit_family_is_not_rejected_by_tuple_count_times_order() -> None:
    degree = 22
    generator = list(range(degree))
    for start, length in ((0, 8), (8, 9), (17, 5)):
        for offset in range(length):
            generator[start + offset] = start + ((offset + 1) % length)
    action = FinitePermutationAction(
        domain=tuple(str(index) for index in range(degree)),
        generators=(tuple(generator),),
    )
    seed = (0, 8, 17)
    family = []
    current = seed
    for _ in range(360):
        family.append(current)
        current = tuple(generator[value] for value in current)
    result = tuple_family_orbit_profile(
        TupleFamilyOrbitSource(action=action, arity=3, family=tuple(family))
    )
    assert len(result.rows) == 1
    assert result.rows[0].orbit_size == 360
    assert result.is_union_of_complete_ambient_orbits is True


def test_complete_prime_cycle_orbit_is_not_rejected_by_images_times_order() -> None:
    degree = 46
    generator = list(range(degree))
    for start, length in ((0, 17), (17, 29)):
        for offset in range(length):
            generator[start + offset] = start + ((offset + 1) % length)
    action = FinitePermutationAction(
        domain=tuple(str(index) for index in range(degree)),
        generators=(tuple(generator),),
    )
    seed = (0, 17)
    family = []
    current = seed
    for _ in range(493):
        family.append(current)
        current = tuple(generator[value] for value in current)
    result = tuple_family_orbit_profile(
        TupleFamilyOrbitSource(action=action, arity=2, family=tuple(family))
    )
    assert len(result.rows) == 1
    assert result.rows[0].orbit_size == 493
    assert result.is_union_of_complete_ambient_orbits is True


def test_negative_source_indices_are_rejected_before_family_lookup() -> None:
    result = tuple_family_orbit_profile(
        TupleFamilyOrbitSource(action=_swap_action(), arity=1, family=((0,), (1,)))
    )
    payload = result.model_dump()
    payload["rows"][0]["source_indices"] = [-3]
    with pytest.raises(ValidationError) as source_index:
        TupleFamilyOrbitResult.model_validate(payload)
    assert "source_index_out_of_range" in source_index.value.errors()[0]["type"]


def test_duplicate_orbit_representatives_are_rejected() -> None:
    result = tuple_family_orbit_profile(
        TupleFamilyOrbitSource(
            action=FinitePermutationAction(domain=("a", "b"), generators=((0, 1),)),
            arity=1,
            family=((0,), (1,)),
        )
    )
    payload = result.model_dump()
    payload["rows"][1]["representative"] = payload["rows"][0]["representative"]
    with pytest.raises(ValidationError) as duplicate:
        TupleFamilyOrbitResult.model_validate(payload)
    assert "rows_not_canonical" in duplicate.value.errors()[0]["type"]


def test_distinct_source_rows_are_indexed_once_before_orbit_partition() -> None:
    action = FinitePermutationAction(
        domain=tuple(str(index) for index in range(40)),
        generators=(tuple(range(40)),),
    )
    family = tuple((index,) for index in range(40))
    result = tuple_family_orbit_profile(
        TupleFamilyOrbitSource(action=action, arity=1, family=family)
    )
    assert [row.representative for row in result.rows] == list(family)
    assert [row.source_indices for row in result.rows] == [
        (index,) for index in range(40)
    ]
    assert all(row.orbit_size == 1 for row in result.rows)
