"""Diagonal tuple-family orbit profiles."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.groups.actions._models import FinitePermutationAction
from jacobian.math.groups.tuple_orbits._models import TupleFamilyOrbitSource
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
    from jacobian.catalog.models import OperationDomainValidationError

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

