"""Complete bounded character-table release slice."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import CharacterTableRequest
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes


def _partition(
    degree: int, generators: tuple[tuple[int, ...], ...]
) -> GroupConjugacyClassesResult:
    source = PermutationGroup(degree=degree, generators=generators)
    classes = group_conjugacy_classes(degree, [list(row) for row in generators])
    return GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )


def test_s3_complete_table_has_orthogonal_rows_and_degree_squares() -> None:
    table = character_table(_partition(3, ((1, 2, 0), (1, 0, 2))))
    assert [row.degree for row in table.rows] == [1, 1, 2]
    assert table.degree_square_sum == 6
    assert table.partition.source.degree == 3
    assert table.rows[2].values[1].coefficients[0].as_fraction() == 0


def test_trivial_group_uses_one_class_same_carrier() -> None:
    table = character_table(_partition(1, ((0,),)))
    assert len(table.partition.classes) == 1
    assert table.rows[0].degree == 1
    assert table.rows[0].values[0].coefficients[0].as_fraction() == 1


def test_unsupported_complete_group_is_rejected_not_partial() -> None:
    # D8 is neither cyclic nor the supported S3 slice.
    with pytest.raises(OperationDomainValidationError):
        character_table(_partition(4, ((1, 0, 3, 2), (0, 2, 1, 3))))


@pytest.mark.parametrize("order", (31, 59, 60))
def test_large_cyclic_table_is_rejected_before_orthogonality_materialization(
    order: int,
) -> None:
    cycle = (*range(1, order), 0)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        character_table(_partition(order, (cycle,)))
    assert (
        exc_info.value.errors()[0]["type"]
        == "groups.characters.table_work_exceeds_envelope"
    )


def test_small_cyclic_table_is_admitted_by_the_same_aggregate_bound() -> None:
    table = character_table(_partition(7, ((*range(1, 7), 0),)))
    assert len(table.rows) == 7
    assert all(row.degree == 1 for row in table.rows)


def test_partition_survives_request_json_round_trip() -> None:
    request = CharacterTableRequest(partition=_partition(3, ((1, 2, 0), (1, 0, 2))))
    restored = CharacterTableRequest.model_validate_json(request.model_dump_json())
    assert restored == request
