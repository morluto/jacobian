"""Exact character kernels with a direct finite-group element oracle."""

from itertools import permutations

import pytest

import jacobian.math.groups.characters.representation_ring_operations as ring_operations
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import (
    GroupConjugacyClassesResult,
    PermutationGroup,
)
from jacobian.math.groups.characters._models import (
    CharacterKernel,
    CharacterKernelRequest,
    CharacterRingElement,
    CharacterRow,
    CharacterTableResult,
    ConjugacyClassPartition,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.characters.representation_ring_operations import (
    character_kernel,
)
from jacobian.math.groups.operations import group_conjugacy_classes, group_order


def test_model_constructed_kernel_request_is_domain_error() -> None:
    with pytest.raises(OperationDomainValidationError):
        character_kernel(CharacterKernelRequest.model_construct())


def _s3_table():
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
    )
    return character_table(partition)


def _element(table, coordinates):
    return CharacterRingElement(
        table=table, irreducible_multiplicities=tuple(coordinates)
    )


def _direct_s3_kernel(coordinates):
    members = []
    for permutation in permutations(range(3)):
        fixed = sum(permutation[index] == index for index in range(3))
        sign = (
            -1
            if sum(
                permutation[i] > permutation[j]
                for i in range(3)
                for j in range(i + 1, 3)
            )
            % 2
            else 1
        )
        standard = fixed - 1
        degree = coordinates[0] + coordinates[1] + 2 * coordinates[2]
        trace = coordinates[0] + coordinates[1] * sign + coordinates[2] * standard
        if trace == degree:
            members.append(permutation)
    return set(members)


def _elements(group: PermutationGroup):
    return {
        tuple(element)
        for conjugacy_class in group_conjugacy_classes(
            group.degree, [list(generator) for generator in group.generators]
        )
        for element in conjugacy_class
    }


@pytest.mark.parametrize(
    ("coordinates", "expected_order"),
    [((0, 1, 0), 3), ((0, 0, 1), 1), ((1, 0, 0), 6), ((1, 1, 0), 3)],
)
def test_s3_character_kernels_match_elementwise_trace_oracle(
    coordinates, expected_order
) -> None:
    table = _s3_table()
    result = character_kernel(
        CharacterKernelRequest(character=_element(table, coordinates))
    )
    actual = _elements(result.subgroup)
    oracle = _direct_s3_kernel(coordinates)
    assert actual == oracle
    assert len(actual) == expected_order
    assert group_order(result.subgroup) == expected_order
    assert result.ambient_group == table.partition.source
    assert CharacterKernel.model_validate_json(result.model_dump_json()) == result


def test_negative_virtual_character_is_rejected_as_not_an_ordinary_character() -> None:
    table = _s3_table()
    with pytest.raises(OperationDomainValidationError):
        character_kernel(CharacterKernelRequest(character=_element(table, (1, -1, 0))))


def test_kernel_catalog_example_executes_and_retains_ambient_parent() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("character.kernel.compute")
    assert operation is not None
    example = operation.examples[0]
    invocation = invoke_operation(operation.operation_id, example.input, catalog)
    result = CharacterKernel.model_validate(invocation.output)
    assert (
        result.subgroup
        == character_kernel(
            CharacterKernelRequest(character=_element(_s3_table(), (0, 1, 0)))
        ).subgroup
    )
    assert result.ambient_group == _s3_table().partition.source


def test_group_order_admission_precedes_conjugacy_expansion(monkeypatch) -> None:
    degree = 61
    generator = tuple((point + 1) % degree for point in range(degree))
    source = PermutationGroup(degree=degree, generators=(generator,))
    partition = ConjugacyClassPartition.model_construct(
        source=source, classes=((tuple(range(degree)),),)
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
    request = CharacterKernelRequest.model_construct(
        character=CharacterRingElement.model_construct(
            table=table, irreducible_multiplicities=(1,)
        )
    )

    def unexpected_conjugacy_expansion(*args, **kwargs):
        raise AssertionError("admission must precede conjugacy expansion")

    monkeypatch.setattr(
        ring_operations, "group_conjugacy_classes", unexpected_conjugacy_expansion
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_kernel(request)
