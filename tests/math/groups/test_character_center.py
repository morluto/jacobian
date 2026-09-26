"""Exact centers of finite-group characters with a direct S3 matrix oracle."""

import json
from itertools import permutations

import pytest

import jacobian.math.groups.characters.representation_ring_operations as ring_operations
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterCenter,
    CharacterCenterRequest,
    CharacterRingElement,
    CharacterRow,
    CharacterTableResult,
    ConjugacyClassPartition,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.characters.representation_ring_operations import (
    character_center,
)
from jacobian.math.groups.operations import group_conjugacy_classes, group_order


def _s3_table():
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
    )
    return character_table(partition)


def _cyclic5_table():
    source = PermutationGroup(degree=5, generators=((1, 2, 3, 4, 0),))
    classes = group_conjugacy_classes(5, [list(g) for g in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
    )
    return character_table(partition)


def _elements(group: PermutationGroup):
    return {
        tuple(element)
        for conjugacy_class in group_conjugacy_classes(
            group.degree, [list(generator) for generator in group.generators]
        )
        for element in conjugacy_class
    }


def _standard_representation_matrix(permutation):
    """Matrix on x1+x2+x3=0 in the basis (e1-e2, e2-e3)."""
    basis = ((1, -1, 0), (0, 1, -1))
    columns = []
    for vector in basis:
        image = tuple(vector[permutation[index]] for index in range(3))
        columns.append((image[0], -image[2]))
    return (
        (columns[0][0], columns[1][0]),
        (columns[0][1], columns[1][1]),
    )


def _direct_standard_center():
    identity = tuple(range(3))
    result = set()
    for permutation in permutations(range(3)):
        matrix = _standard_representation_matrix(permutation)
        if matrix[0][1] == matrix[1][0] == 0 and matrix[0][0] == matrix[1][1]:
            result.add(permutation)
    assert identity in result
    return result


def _direct_trivial_plus_sign_center():
    result = set()
    for permutation in permutations(range(3)):
        inversions = sum(
            permutation[i] > permutation[j] for i in range(3) for j in range(i + 1, 3)
        )
        sign = -1 if inversions % 2 else 1
        matrix = ((1, 0), (0, sign))
        if matrix[0][1] == matrix[1][0] == 0 and matrix[0][0] == matrix[1][1]:
            result.add(permutation)
    return result


def _fifth_root_coefficients(exponent):
    exponent %= 5
    if exponent == 4:
        return (-1, -1, -1, -1)
    return tuple(int(power == exponent) for power in range(4))


def _compute(coordinates):
    table = _s3_table()
    character = CharacterRingElement(
        table=table, irreducible_multiplicities=tuple(coordinates)
    )
    return table, character_center(CharacterCenterRequest(character=character))


def test_s3_standard_character_center_matches_matrix_representation_oracle():
    table, result = _compute((0, 0, 1))
    assert _elements(result.subgroup) == _direct_standard_center()
    assert len(_direct_standard_center()) == 1
    assert result.scalar_class_indices == (0,)
    assert result.scalar_values[0].coefficients[0].as_fraction() == 1
    assert result.character.table == table
    assert CharacterCenter.model_validate_json(result.model_dump_json()) == result


def test_s3_reducible_character_finds_scalar_a3_action():
    _, result = _compute((1, 1, 0))
    assert _elements(result.subgroup) == _direct_trivial_plus_sign_center()
    assert len(_elements(result.subgroup)) == 3
    assert group_order(result.subgroup) == 3
    assert len(result.scalar_class_indices) == len(result.scalar_values) == 2
    assert all(
        value.coefficients[0].as_fraction() == 1 for value in result.scalar_values
    )


def test_one_dimensional_sign_character_is_scalar_on_the_whole_group():
    table, result = _compute((0, 1, 0))
    assert _elements(result.subgroup) == _elements(table.partition.source)
    assert group_order(result.subgroup) == 6
    assert len(result.scalar_class_indices) == 3
    assert {value.coefficients[0].as_fraction() for value in result.scalar_values} == {
        -1,
        1,
    }


def test_cyclic_fifth_root_character_keeps_exact_cyclotomic_scalar_values():
    table = _cyclic5_table()
    coordinates = tuple(int(index == 1) for index in range(5))
    character = CharacterRingElement(
        table=table, irreducible_multiplicities=coordinates
    )
    result = character_center(CharacterCenterRequest(character=character))
    source_group = table.partition.source
    assert _elements(result.subgroup) == _elements(source_group)
    assert result.scalar_class_indices == tuple(range(5))

    generator = source_group.generators[0]
    powers = [tuple(range(source_group.degree))]
    for _ in range(4):
        previous = powers[-1]
        powers.append(tuple(previous[generator[index]] for index in range(5)))
    expected_by_representative = {
        element: _fifth_root_coefficients(exponent)
        for exponent, element in enumerate(powers)
    }
    actual_coefficients = tuple(
        tuple(coefficient.as_fraction() for coefficient in value.coefficients)
        for value in result.scalar_values
    )
    expected_coefficients = tuple(
        expected_by_representative[conjugacy_class[0]]
        for conjugacy_class in table.partition.classes
    )
    assert actual_coefficients == expected_coefficients


def test_model_constructed_request_missing_character_is_domain_error():
    with pytest.raises(OperationDomainValidationError) as exc_info:
        character_center(CharacterCenterRequest.model_construct())
    assert exc_info.value.errors()[0]["type"] == "groups.characters.center_input_type"


def test_negative_virtual_character_is_rejected():
    table = _s3_table()
    character = CharacterRingElement(table=table, irreducible_multiplicities=(1, -1, 0))
    with pytest.raises(OperationDomainValidationError):
        character_center(CharacterCenterRequest(character=character))


def test_center_catalog_example_executes():
    catalog = Catalog.open()
    operation = catalog.operation("character.center.compute")
    assert operation is not None
    invocation = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    result = CharacterCenter.model_validate_json(json.dumps(invocation.output))
    assert _elements(result.subgroup) == _direct_standard_center()


def test_character_center_rejects_zero_dimension_ordinary_value():
    table = _s3_table()
    zero = CharacterRingElement(table=table, irreducible_multiplicities=(0, 0, 0))
    with pytest.raises(OperationDomainValidationError):
        character_center(CharacterCenterRequest(character=zero))


def test_order_bound_precedes_conjugacy_expansion(monkeypatch):
    degree = 61
    generator = tuple((point + 1) % degree for point in range(degree))
    source = PermutationGroup(degree=degree, generators=(generator,))
    identity = tuple(range(degree))
    partition = ConjugacyClassPartition.model_construct(
        source=source, classes=((identity,),)
    )
    base = _s3_table()
    table = CharacterTableResult.model_construct(
        partition=partition,
        axis=base.axis,
        rows=(
            CharacterRow.model_construct(
                label="trivial", degree=1, values=(base.rows[0].values[0],)
            ),
        ),
        degree_square_sum=1,
    )
    character = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(1,)
    )

    def unexpected_conjugacy_expansion(*args, **kwargs):
        raise AssertionError("admission must precede conjugacy expansion")

    monkeypatch.setattr(
        ring_operations, "group_conjugacy_classes", unexpected_conjugacy_expansion
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_center(CharacterCenterRequest.model_construct(character=character))


def test_output_admission_accounts_for_retained_full_character_table():
    table = _cyclic5_table()
    character = CharacterRingElement(
        table=table, irreducible_multiplicities=(1, 0, 0, 0, 0)
    )
    # Admission remains comfortably inside the envelope for a small table,
    # while the quadratic class-table term is included before expansion.
    result = character_center(CharacterCenterRequest(character=character))
    assert group_order(result.subgroup) == 5
    assert len(result.character.table.partition.classes) == 5
