"""Ordinary second indicators checked against elementwise squaring."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._models import GroupConjugacyClassesRequest
from jacobian.math.groups._tools import compute_group_conjugacy_classes
from jacobian.math.groups.characters.operations import (
    character_table,
    frobenius_schur_indicator,
)


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(right[left[index]] for index in range(len(left)))


def _square(value: tuple[int, ...]) -> tuple[int, ...]:
    return _compose(value, value)


@pytest.mark.parametrize(
    ("degree", "generators", "row_index", "expected"),
    [
        (3, ((1, 2, 0), (1, 0, 2)), 2, 1),  # real standard representation of S3
        (3, ((1, 2, 0),), 1, 0),  # nonreal linear character of C3
    ],
)
def test_second_indicator_matches_elementwise_square_oracle(
    degree: int,
    generators: tuple[tuple[int, ...], ...],
    row_index: int,
    expected: int,
) -> None:
    partition = compute_group_conjugacy_classes(
        GroupConjugacyClassesRequest(degree=degree, generators=generators)
    )
    table = character_table(partition)
    result = frobenius_schur_indicator(table, row_index)

    class_of = {
        tuple(element): class_index
        for class_index, conjugacy_class in enumerate(table.partition.classes)
        for element in conjugacy_class
    }
    row = table.rows[row_index]
    # The independent oracle sums over concrete elements instead of class weights.
    total = [Fraction(0) for _ in row.values[0].coefficients]
    for element in class_of:
        coefficients = row.values[class_of[_square(element)]].coefficients
        for index, coefficient in enumerate(coefficients):
            total[index] += coefficient.as_fraction()
    total = [coefficient / table.axis.group_order for coefficient in total]
    assert total == [Fraction(expected), *([Fraction(0)] * (len(total) - 1))]
    assert result.indicator == expected
    assert result.source_table == table
    assert result.row_index == row_index


def test_second_indicator_rejects_a_forged_incomplete_character_table() -> None:
    partition = compute_group_conjugacy_classes(
        GroupConjugacyClassesRequest(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    )
    table = character_table(partition)
    changed_row = table.rows[2].model_copy(
        update={"values": (table.rows[2].values[0],) * 3}
    )
    forged = table.model_copy(update={"rows": (*table.rows[:2], changed_row)})
    with pytest.raises(
        OperationDomainValidationError, match="complete canonical table"
    ):
        frobenius_schur_indicator(forged, 2)


def test_second_indicator_rejects_nonexistent_row() -> None:
    partition = compute_group_conjugacy_classes(
        GroupConjugacyClassesRequest(degree=3, generators=((1, 2, 0),))
    )
    table = character_table(partition)
    with pytest.raises(OperationDomainValidationError, match="row_index"):
        frobenius_schur_indicator(table, 3)


def test_catalog_example_is_runnable() -> None:
    from jacobian.math.groups.characters._tools import TOOLS

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "group.character.frobenius_schur_indicator.compute"
    )
    import json

    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    assert tool.run(request).indicator == 1


def test_native_indicator_rejects_malformed_inputs_without_pydantic_leak() -> None:
    partition = compute_group_conjugacy_classes(
        GroupConjugacyClassesRequest(degree=3, generators=((1, 2, 0),))
    )
    table = character_table(partition)
    with pytest.raises(OperationDomainValidationError, match="row_index"):
        frobenius_schur_indicator(table, -1)
    with pytest.raises(OperationDomainValidationError, match="row_index"):
        frobenius_schur_indicator(table, True)
    with pytest.raises(OperationDomainValidationError, match="table"):
        frobenius_schur_indicator({"partition": None}, 0)
