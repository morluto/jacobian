from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    ClassMultiplicationConstantsRequest,
    ClassMultiplicationConstantsResult,
)
from jacobian.math.groups.characters.class_algebra import (
    _admit_class_algebra_size,
    class_multiplication_constants,
)
from jacobian.math.groups.operations import group_conjugacy_classes


def _partition(generators: list[list[int]]) -> GroupConjugacyClassesResult:
    degree = len(generators[0])
    classes = group_conjugacy_classes(degree, generators)
    return GroupConjugacyClassesResult(
        source=PermutationGroup(
            degree=degree,
            generators=tuple(tuple(generator) for generator in generators),
        ),
        classes=tuple(
            tuple(tuple(member) for member in conjugacy_class)
            for conjugacy_class in classes
        ),
    )


def test_s3_class_sum_products_match_exact_hand_values() -> None:
    # Class order: identity, transpositions, 3-cycles.
    partition = _partition([[1, 2, 0], [1, 0, 2]])

    result = class_multiplication_constants(
        ClassMultiplicationConstantsRequest(partition=partition)
    )

    assert result.constants == (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (3, 0, 3), (0, 2, 0)),
        ((0, 0, 1), (0, 2, 0), (2, 0, 1)),
    )
    sizes = tuple(len(conjugacy_class) for conjugacy_class in partition.classes)
    for left in range(len(sizes)):
        for right in range(len(sizes)):
            assert (
                sum(
                    result.constants[left][right][target] * sizes[target]
                    for target in range(len(sizes))
                )
                == sizes[left] * sizes[right]
            )


def test_s3_catalog_example_publishes_the_complete_tensor() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("group.class_multiplication_constants.compute")
    assert operation is not None
    example = operation.examples[0]

    invocation = invoke_operation(operation.operation_id, example.input, catalog)
    result = ClassMultiplicationConstantsResult.model_validate(invocation.output)

    assert result.constants[1][1] == (3, 0, 3)
    assert result.constants[2][2] == (2, 0, 1)


def test_cyclic_class_algebra_records_group_multiplication() -> None:
    partition = _partition([[1, 2, 0]])

    result = class_multiplication_constants(
        ClassMultiplicationConstantsRequest(partition=partition)
    )

    assert result.constants == (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    )


def test_rejects_partition_not_bound_to_its_source_group() -> None:
    claim = GroupConjugacyClassesResult(
        source=PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2))),
        classes=(((0, 1, 2),),),
    )

    with pytest.raises(OperationDomainValidationError):
        class_multiplication_constants(
            ClassMultiplicationConstantsRequest(partition=claim)
        )


def test_actual_group_order_bound_precedes_conjugacy_expansion() -> None:
    source = PermutationGroup(
        degree=7,
        generators=((1, 2, 3, 4, 5, 6, 0), (1, 0, 2, 3, 4, 5, 6)),
    )
    forged_small_partition = GroupConjugacyClassesResult(
        source=source,
        classes=(((0, 1, 2, 3, 4, 5, 6),),),
    )

    with pytest.raises(OperationResourceAdmissionError, match="order at most 256"):
        class_multiplication_constants(
            ClassMultiplicationConstantsRequest(partition=forged_small_partition)
        )


def test_maximum_tensor_shape_is_admitted_without_serialized_size_gate() -> None:
    # A concrete cyclic group reaches the operation's maximum tensor-cell
    # boundary and produces the complete, exact class algebra.
    generator = (*range(1, 64), 0)
    cyclic_64 = _partition([list(generator)])
    result = class_multiplication_constants(
        ClassMultiplicationConstantsRequest(partition=cyclic_64)
    )
    assert len(result.constants) == 64
    assert (
        sum(
            len(pair_coefficients)
            for plane in result.constants
            for pair_coefficients in plane
        )
        == 64**3
    )
    square = tuple(generator[generator[index]] for index in range(64))
    class_index = {
        conjugacy_class[0]: index
        for index, conjugacy_class in enumerate(result.partition.classes)
    }
    assert (
        result.constants[class_index[generator]][class_index[generator]][
            class_index[square]
        ]
        == 1
    )

    # This shape is within the public mathematical envelope: 64^3 exact
    # integer coefficients, with group-order-bounded coefficient height. The
    # previous JSON-size estimate rejected it even though the cell, work, and
    # scalar bounds admit the result.
    _admit_class_algebra_size(order=128, class_count=64, degree=8)
