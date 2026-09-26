"""Exact checks for the degree of finite-group characters."""

from __future__ import annotations

import json

import pytest

import jacobian.math.groups.characters.degree.operations as degree_operations
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterTableResult,
)
from jacobian.math.groups.characters.degree import (
    CharacterDegree,
    CharacterDegreeRequest,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes

OPERATION_ID = "character.degree.compute"


def _s3_table() -> CharacterTableResult:
    generators = ((1, 2, 0), (1, 0, 2))
    group = PermutationGroup(degree=3, generators=generators)
    classes = group_conjugacy_classes(3, [list(row) for row in generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        group,
        tuple(tuple(tuple(element) for element in cls) for cls in classes),
    )
    return character_table(partition)


def test_s3_character_degree_matches_independent_identity_value() -> None:
    table = _s3_table()
    multiplicities = (2, 1, 3)
    character = CharacterRingElement(
        table=table,
        irreducible_multiplicities=multiplicities,
    )

    result = degree_operations.character_degree(
        CharacterDegreeRequest(character=character)
    )

    # The S3 irreducibles have degrees 1, 1, and 2; direct evaluation at the
    # identity permutation independently gives the same value.
    direct_identity_value = (
        multiplicities[0] + multiplicities[1] + 2 * multiplicities[2]
    )
    assert result.degree == direct_identity_value == 9
    assert result.character == character


def test_irreducible_and_reducible_degrees_are_exact() -> None:
    table = _s3_table()
    assert [
        degree_operations.character_degree(
            CharacterDegreeRequest(
                character=CharacterRingElement(
                    table=table,
                    irreducible_multiplicities=tuple(int(i == row) for i in range(3)),
                )
            )
        ).degree
        for row in range(3)
    ] == [1, 1, 2]


def test_virtual_character_is_not_misreported_as_an_ordinary_degree() -> None:
    character = CharacterRingElement(
        table=_s3_table(), irreducible_multiplicities=(1, -1, 0)
    )
    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(CharacterDegreeRequest(character=character))
    assert error.value.errors()[0]["type"] == (
        "groups.characters.degree_requires_ordinary_character"
    )


def test_forged_character_table_is_rejected_after_source_authentication() -> None:
    table = _s3_table()
    rows = (table.rows[0].model_copy(update={"degree": 2}), *table.rows[1:])
    forged_table = table.model_copy(update={"rows": rows})
    forged_character = CharacterRingElement.model_construct(
        table=forged_table,
        irreducible_multiplicities=(1, 0, 0),
    )

    with pytest.raises(OperationDomainValidationError) as error:
        degree_operations.character_degree(
            CharacterDegreeRequest.model_construct(character=forged_character)
        )
    assert error.value.errors()[0]["type"] == (
        "groups.characters.degree_noncanonical_table"
    )


def test_unsupported_source_order_rejects_before_class_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _s3_table()
    cycle = (*range(1, 61), 0)
    unsupported = PermutationGroup(degree=61, generators=(cycle,))
    extended_classes = tuple(
        tuple(element + tuple(range(3, 61)) for element in cls)
        for cls in table.partition.classes
    )
    partition = table.partition.model_copy(
        update={"source": unsupported, "classes": extended_classes}
    )
    axis = table.axis.model_copy(
        update={
            "group": unsupported,
            "class_representatives": tuple(cls[0] for cls in extended_classes),
        }
    )
    forged_table = table.model_copy(update={"partition": partition, "axis": axis})
    character = CharacterRingElement.model_construct(
        table=forged_table,
        irreducible_multiplicities=(1, 0, 0),
    )
    request = CharacterDegreeRequest.model_construct(character=character)

    def fail_if_expanded(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unsupported source order must reject before class expansion")

    monkeypatch.setattr(degree_operations, "group_conjugacy_classes", fail_if_expanded)
    with pytest.raises(OperationResourceAdmissionError):
        degree_operations.character_degree(request)


def test_catalog_example_dispatch_and_result_round_trip() -> None:
    catalog = Catalog.open()
    tool = catalog.operation(OPERATION_ID)
    assert tool is not None
    assert tool.examples

    response = invoke_operation(OPERATION_ID, tool.examples[0].input, catalog)
    result = CharacterDegree.model_validate_json(json.dumps(response.output))

    assert result.degree == 3
    assert result.character.irreducible_multiplicities == (3,)
