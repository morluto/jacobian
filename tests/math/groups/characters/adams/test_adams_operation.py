"""Independent finite-group oracle for exact Adams operations."""

import json
from itertools import permutations

import pytest
from pydantic import ValidationError

import jacobian.math.groups.characters.adams.operations as adams_operations
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import CharacterRingElement
from jacobian.math.groups.characters.adams._models import AdamsOperationRequest
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes


def _s3_table(*, degree: int = 3):
    generators = tuple(
        tuple(list(generator) + list(range(3, degree)))
        for generator in ((1, 2, 0), (1, 0, 2))
    )
    group = PermutationGroup(
        degree=degree,
        generators=generators,
    )
    classes = group_conjugacy_classes(
        degree, [list(generator) for generator in generators]
    )
    partition = GroupConjugacyClassesResult._from_kernel(
        group,
        tuple(tuple(tuple(element) for element in cls) for cls in classes),
    )
    return character_table(partition)


def _cyclic_three_table():
    group = PermutationGroup(degree=3, generators=((1, 2, 0),))
    classes = group_conjugacy_classes(3, [[1, 2, 0]])
    partition = GroupConjugacyClassesResult._from_kernel(
        group,
        tuple(tuple(tuple(element) for element in cls) for cls in classes),
    )
    return character_table(partition)


def _sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(left[right[index]] for index in range(len(left)))


def _power(permutation: tuple[int, ...], exponent: int) -> tuple[int, ...]:
    result = tuple(range(len(permutation)))
    for _ in range(exponent):
        result = _compose(result, permutation)
    return result


def _oracle_value(permutation: tuple[int, ...], coordinates: tuple[int, ...]) -> int:
    # S3 irreducible characters: 1, sign, and the standard trace (#fixed - 1).
    fixed = sum(permutation[index] == index for index in range(3))
    return (
        coordinates[0]
        + coordinates[1] * _sign(permutation)
        + coordinates[2] * (fixed - 1)
    )


def _irreducible_value(permutation: tuple[int, ...], index: int) -> int:
    if index == 0:
        return 1
    if index == 1:
        return _sign(permutation)
    return sum(permutation[point] == point for point in range(3)) - 1


@pytest.fixture(scope="module")
def s3_table():
    return _s3_table()


@pytest.fixture(scope="module")
def cyclic_three_table():
    return _cyclic_three_table()


def test_s3_adams_coordinates_match_direct_element_enumeration(s3_table) -> None:
    table = s3_table
    source_coordinates = (2, -1, 1)
    source = CharacterRingElement(
        table=table,
        irreducible_multiplicities=source_coordinates,
    )
    exponent = 2
    catalog = Catalog.open()
    assert catalog.operation("character.adams_operation.compute") is not None
    result = invoke_operation(
        "character.adams_operation.compute",
        AdamsOperationRequest(character=source, exponent=exponent).model_dump(
            mode="json"
        ),
        catalog,
    )
    ring_element = CharacterRingElement.model_validate_json(json.dumps(result.output))

    elements = tuple(permutations(range(3)))
    expected_coordinates = tuple(
        sum(
            _oracle_value(_power(element, exponent), source_coordinates)
            * _irreducible_value(element, index)
            for element in elements
        )
        // 6
        for index in range(3)
    )
    assert ring_element.irreducible_multiplicities == expected_coordinates
    assert ring_element.table == table
    assert (
        CharacterRingElement.model_validate_json(ring_element.model_dump_json())
        == ring_element
    )


def test_large_cyclic_three_adams_exponent_is_trivial(cyclic_three_table) -> None:
    table = cyclic_three_table
    nontrivial_row = CharacterRingElement(
        table=table,
        irreducible_multiplicities=(0, 1, 0),
    )
    result = invoke_operation(
        "character.adams_operation.compute",
        AdamsOperationRequest(
            character=nontrivial_row, exponent=10**100 + 2
        ).model_dump(mode="json"),
        Catalog.open(),
    )

    assert CharacterRingElement.model_validate_json(json.dumps(result.output)) == (
        CharacterRingElement(
            table=table,
            irreducible_multiplicities=(1, 0, 0),
        )
    )


def test_adams_exponent_contract_rejects_zero(s3_table) -> None:
    table = s3_table
    character = CharacterRingElement(table=table, irreducible_multiplicities=(1, 0, 0))
    with pytest.raises(ValidationError):
        AdamsOperationRequest(character=character, exponent=0)


def test_adams_work_bound_rejects_huge_bit_length_before_class_expansion(
    monkeypatch,
) -> None:
    wide_table = _s3_table(degree=64)
    wide_character = CharacterRingElement(
        table=wide_table,
        irreducible_multiplicities=(0, 0, 1),
    )
    request = AdamsOperationRequest.model_construct(
        character=wide_character,
        exponent=10**32_767 + 2,
    )

    def fail_if_expanded(*args, **kwargs):
        pytest.fail("admission must reject before conjugacy-class expansion")

    monkeypatch.setattr(adams_operations, "group_conjugacy_classes", fail_if_expanded)
    with pytest.raises(OperationResourceAdmissionError):
        adams_operations.character_adams_operation(request)
