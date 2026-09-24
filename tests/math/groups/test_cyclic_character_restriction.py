"""Exact restriction from supported cyclic groups to their unique subgroups."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters import operations as character_operations
from jacobian.math.groups.characters._models import CyclicCharacterRestrictionRequest
from jacobian.math.groups.characters.operations import restrict_cyclic_character
from jacobian.math.groups.operations import group_conjugacy_classes


def _partition(order: int, degree: int | None = None) -> GroupConjugacyClassesResult:
    degree = order if degree is None else degree
    generator = (*range(1, order), 0, *range(order, degree))
    source = PermutationGroup(degree=degree, generators=(generator,))
    classes = group_conjugacy_classes(degree, [list(generator)])
    return GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )


def _restrict(order: int, row: int, subgroup_order: int):
    return restrict_cyclic_character(
        CyclicCharacterRestrictionRequest(
            partition=_partition(order), row_index=row, subgroup_order=subgroup_order
        )
    )


def test_c4_irreducible_restrictions_to_unique_c2_keep_embedding_and_values() -> None:
    result = _restrict(4, 1, 2)
    assert result.source_table.rows[1].label == "chi_1"
    assert result.target_partition.source.degree == 4
    assert tuple(len(row) for row in result.target_partition.classes) == (1, 1)
    assert result.source_class_indices == (0, 2)
    assert tuple(
        value.coefficients[0].as_fraction()
        for value in result.restricted_character.values
    ) == (Fraction(1), Fraction(-1))
    assert result.restricted_character.axis.cyclotomic_order == 4


def test_c6_restriction_to_unique_c3_preserves_nontrivial_exact_values() -> None:
    result = _restrict(6, 1, 3)
    assert result.source_class_indices == (0, 2, 4)
    assert tuple(value.order for value in result.restricted_character.values) == (
        6,
        6,
        6,
    )
    assert result.restricted_character.values == tuple(
        result.source_table.rows[1].values[index]
        for index in result.source_class_indices
    )
    assert result.target_partition.classes == tuple(
        (result.target_partition.classes[index][0],) for index in range(3)
    )


def test_restriction_handles_identity_subgroup() -> None:
    result = _restrict(5, 3, 1)
    assert len(result.target_partition.classes) == 1
    assert result.source_class_indices == (0,)
    assert result.restricted_character.values == (
        result.source_table.rows[3].values[0],
    )


@pytest.mark.parametrize(
    ("row_index", "subgroup_order"),
    [(True, 1), (0, True)],
)
def test_restriction_request_rejects_boolean_integer_fields(
    row_index: int, subgroup_order: int
) -> None:
    with pytest.raises(ValidationError):
        CyclicCharacterRestrictionRequest.model_validate(
            {
                "partition": _partition(4),
                "row_index": row_index,
                "subgroup_order": subgroup_order,
            }
        )


@pytest.mark.parametrize("order,subgroup_order", [(6, 4), (6, 0)])
def test_restriction_rejects_nondivisor_or_zero_subgroup(
    order: int, subgroup_order: int
) -> None:
    with pytest.raises((OperationDomainValidationError, ValueError)):
        _restrict(order, 0, subgroup_order)


def test_restriction_does_not_generalize_to_noncyclic_table_families() -> None:
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(3, [list(row) for row in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )
    with pytest.raises(OperationDomainValidationError):
        restrict_cyclic_character(
            CyclicCharacterRestrictionRequest(
                partition=partition, row_index=0, subgroup_order=3
            )
        )


def test_degree_aware_maximum_work_is_admitted_and_just_over_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = _partition(60, degree=64)
    work = character_operations._cyclic_restriction_work(60, 64, 60)
    assert work <= character_operations.MAX_CYCLIC_RESTRICTION_WORK
    request = CyclicCharacterRestrictionRequest(
        partition=partition, row_index=59, subgroup_order=60
    )

    monkeypatch.setattr(character_operations, "MAX_CYCLIC_RESTRICTION_WORK", work - 1)
    with pytest.raises(OperationResourceAdmissionError) as raised:
        restrict_cyclic_character(request)
    assert raised.value.errors()[0]["type"] == (
        "groups.characters.restriction_work_exceeds_envelope"
    )

    monkeypatch.setattr(character_operations, "MAX_CYCLIC_RESTRICTION_WORK", work)
    result = restrict_cyclic_character(request)
    assert len(result.target_partition.classes) == 60
    assert len(result.source_class_indices) == 60
    assert len(result.restricted_character.values) == 60
