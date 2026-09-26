"""Exact tensor-product decomposition for the canonical S3 table."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import CharacterTensorDecompositionRequest
from jacobian.math.groups.characters.operations import character_tensor_decomposition
from jacobian.math.groups.operations import group_conjugacy_classes


def _s3_partition() -> GroupConjugacyClassesResult:
    degree = 3
    generators = ((1, 2, 0), (1, 0, 2))
    source = PermutationGroup(degree=degree, generators=generators)
    classes = group_conjugacy_classes(degree, [list(row) for row in generators])
    return GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )


def test_standard_tensor_square_matches_independent_s3_decomposition() -> None:
    table_values = ((1, 1, 1), (1, -1, 1), (2, 0, -1))
    expected_multiplicities = {
        (0, 0): (1, 0, 0),
        (0, 1): (0, 1, 0),
        (0, 2): (0, 0, 1),
        (1, 0): (0, 1, 0),
        (1, 1): (1, 0, 0),
        (1, 2): (0, 0, 1),
        (2, 0): (0, 0, 1),
        (2, 1): (0, 0, 1),
        (2, 2): (1, 1, 1),
    }
    for (left, right), expected in expected_multiplicities.items():
        result = character_tensor_decomposition(
            CharacterTensorDecompositionRequest(
                partition=_s3_partition(),
                left_row_index=left,
                right_row_index=right,
            )
        )
        assert result.table.axis.class_sizes == (1, 3, 2)
        assert result.tensor_product.axis == result.table.axis
        assert tuple(
            value.coefficients[0].as_fraction()
            for value in result.tensor_product.values
        ) == tuple(
            Fraction(a * b)
            for a, b in zip(table_values[left], table_values[right], strict=True)
        )
        assert result.multiplicities == expected


def test_regular_degree_six_s3_partition_decomposes_in_its_class_order() -> None:
    degree = 6
    generators = ((1, 4, 5, 2, 0, 3), (3, 5, 4, 0, 2, 1))
    source = PermutationGroup(degree=degree, generators=generators)
    classes = group_conjugacy_classes(degree, [list(row) for row in generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )
    assert tuple(map(len, partition.classes)) == (1, 2, 3)

    result = character_tensor_decomposition(
        CharacterTensorDecompositionRequest(
            partition=partition, left_row_index=2, right_row_index=2
        )
    )
    assert result.table.axis.class_sizes == (1, 2, 3)
    assert result.multiplicities == (1, 1, 1)
    assert tuple(
        value.coefficients[0].as_fraction() for value in result.tensor_product.values
    ) == (Fraction(4), Fraction(1), Fraction(0))


def test_sign_tensor_standard_is_standard() -> None:
    result = character_tensor_decomposition(
        CharacterTensorDecompositionRequest(
            partition=_s3_partition(), left_row_index=1, right_row_index=2
        )
    )
    assert result.multiplicities == (0, 0, 1)
    assert [
        value.coefficients[0].as_fraction() for value in result.tensor_product.values
    ] == [
        Fraction(2),
        Fraction(0),
        Fraction(-1),
    ]


def test_tensor_decomposition_rejects_a7_before_conjugacy_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These generators give A7 (order 2520), so re-authenticating its partition
    # expands thousands of permutations although tensor decomposition is S3-only.
    degree = 7
    generators = ((1, 2, 3, 4, 5, 6, 0), (1, 2, 0, 3, 4, 5, 6))
    source = PermutationGroup(degree=degree, generators=generators)
    classes = group_conjugacy_classes(degree, [list(row) for row in generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )

    def unexpected_expansion(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported A7 partition reached conjugacy expansion")

    monkeypatch.setattr(
        "jacobian.math.groups.operations.group_conjugacy_classes",
        unexpected_expansion,
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        character_tensor_decomposition(
            CharacterTensorDecompositionRequest(
                partition=partition, left_row_index=0, right_row_index=0
            )
        )
    assert exc_info.value.errors()[0]["type"] == (
        "groups.characters.tensor_group_unsupported"
    )


def test_tensor_decomposition_rejects_other_supported_table_families() -> None:
    source = PermutationGroup(degree=3, generators=((1, 0, 2),))
    classes = group_conjugacy_classes(3, [[1, 0, 2]])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(member) for member in row) for row in classes)
    )
    with pytest.raises(OperationDomainValidationError):
        character_tensor_decomposition(
            CharacterTensorDecompositionRequest(
                partition=partition, left_row_index=0, right_row_index=0
            )
        )


def test_forged_tensor_request_raises_domain_error() -> None:
    forged = CharacterTensorDecompositionRequest.model_construct(
        partition=_s3_partition(), left_row_index=-1, right_row_index=0
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        character_tensor_decomposition(forged)
    assert exc_info.value.errors()[0]["type"] == (
        "groups.characters.tensor_request_invalid"
    )


def test_tensor_decomposition_rejects_row_index_outside_basis() -> None:
    with pytest.raises(OperationDomainValidationError):
        character_tensor_decomposition(
            CharacterTensorDecompositionRequest(
                partition=_s3_partition(), left_row_index=3, right_row_index=0
            )
        )


def test_catalog_example_executes() -> None:
    catalog = Catalog.open()
    operation = catalog.operation(
        "finite_group.character.tensor_product.decompose.compute"
    )
    assert operation is not None
    result = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert result.output["multiplicities"] == [1, 1, 1]
