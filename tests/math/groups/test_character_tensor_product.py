"""Exact tensor products in supported finite representation rings."""

from itertools import permutations

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterRow,
    CharacterTableResult,
    CharacterTensorProductRequest,
    ConjugacyClassPartition,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.characters.representation_ring_operations import (
    character_tensor_product,
)
from jacobian.math.groups.operations import group_conjugacy_classes


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


def _direct_s3_tensor_square_multiplicities() -> tuple[int, int, int]:
    # The standard module is the quotient of C^3 by the invariant line.
    # Thus its trace at a permutation is fixed_points(g)-1. Tensor traces
    # square pointwise; exact averaging over the six represented elements
    # gives multiplicities against the independently described irreducibles.
    totals = [0, 0, 0]
    for p in permutations(range(3)):
        fixed = sum(p[i] == i for i in range(3))
        standard = fixed - 1
        sign = (
            -1 if sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) % 2 else 1
        )
        for index, irreducible in enumerate((1, sign, standard)):
            totals[index] += standard * standard * irreducible
    return tuple(value // 6 for value in totals)


def test_s3_standard_square_matches_direct_group_representation_oracle() -> None:
    table = _s3_table()
    result = character_tensor_product(
        CharacterTensorProductRequest(
            left=_element(table, (0, 0, 1)),
            right=_element(table, (0, 0, 1)),
        )
    )
    assert result.irreducible_multiplicities == (1, 1, 1)
    assert (
        result.irreducible_multiplicities == _direct_s3_tensor_square_multiplicities()
    )
    assert CharacterRingElement.model_validate_json(result.model_dump_json()) == result


def test_cyclic_and_virtual_tensor_products_are_exact() -> None:
    source = PermutationGroup(degree=3, generators=((1, 2, 0),))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    table = character_table(
        GroupConjugacyClassesResult._from_kernel(
            source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
        )
    )
    # The row ordering is the canonical irreducible order. Multiplication by
    # the inverse one-dimensional character shifts the C3 character index.
    result = character_tensor_product(
        CharacterTensorProductRequest(
            left=_element(table, (0, 1, 0)),
            right=_element(table, (0, 0, 1)),
        )
    )
    assert result.irreducible_multiplicities == (1, 0, 0)
    virtual = character_tensor_product(
        CharacterTensorProductRequest(
            left=_element(table, (-1, 1, 0)),
            right=_element(table, (1, 0, 0)),
        )
    )
    assert virtual.irreducible_multiplicities == (-1, 1, 0)


def test_parent_mismatch_and_noncanonical_table_rejected() -> None:
    table = _s3_table()
    source = PermutationGroup(degree=3, generators=((1, 2, 0),))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    other = character_table(
        GroupConjugacyClassesResult._from_kernel(
            source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
        )
    )
    with pytest.raises(OperationDomainValidationError):
        character_tensor_product(
            CharacterTensorProductRequest(
                left=_element(table, (1, 0, 0)), right=_element(other, (1, 0, 0))
            )
        )
    forged = table.model_copy(
        update={
            "rows": (
                CharacterRow.model_construct(
                    label="forged", degree=1, values=table.rows[0].values
                ),
                *table.rows[1:],
            )
        }
    )
    with pytest.raises(OperationDomainValidationError):
        character_tensor_product(
            CharacterTensorProductRequest.model_construct(
                left=CharacterRingElement.model_construct(
                    table=forged, irreducible_multiplicities=(1, 0, 0)
                ),
                right=_element(table, (1, 0, 0)),
            )
        )


def test_operation_is_published_and_has_exact_result_type() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "character.tensor_product.compute"
    )
    assert tool.request_type is CharacterTensorProductRequest
    assert tool.result_type is CharacterRingElement
    assert "cyclic groups through order 60" in tool.description


def test_oversized_model_constructed_coordinates_rejected_before_group_work(
    monkeypatch,
) -> None:
    table = _s3_table()
    oversized = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(10**513, 0, 0)
    )
    request = CharacterTensorProductRequest.model_construct(
        left=oversized,
        right=_element(table, (1, 0, 0)),
    )

    def unexpected_group_expansion(*args, **kwargs):
        raise AssertionError("group order must not run before raw input admission")

    monkeypatch.setattr(
        "jacobian.math.groups.characters.representation_ring_operations.group_order",
        unexpected_group_expansion,
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_tensor_product(request)


def test_huge_model_constructed_integer_never_reaches_decimal_conversion() -> None:
    table = _s3_table()
    huge = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(10**5000, 0, 0)
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_tensor_product(
            CharacterTensorProductRequest.model_construct(
                left=huge, right=_element(table, (1, 0, 0))
            )
        )


def test_forged_order_one_partition_cannot_trigger_large_group_order_call(
    monkeypatch,
) -> None:
    # A 64-cycle and a reflection generate D_64, of order 128. The forged
    # carrier claims a one-element partition, so admission must derive its
    # backend-work bound from the source permutations alone.
    degree = 64
    rotation = tuple((point + 1) % degree for point in range(degree))
    reflection = tuple((-point) % degree for point in range(degree))
    source = PermutationGroup(degree=degree, generators=(rotation, reflection))
    small_table = _s3_table()
    identity = tuple(range(degree))
    forged_partition = ConjugacyClassPartition.model_construct(
        source=source, classes=((identity,),)
    )
    forged_table = CharacterTableResult.model_construct(
        partition=forged_partition,
        axis=small_table.axis,
        rows=(
            CharacterRow.model_construct(
                label="trivial", degree=1, values=(small_table.rows[0].values[0],)
            ),
        ),
        degree_square_sum=1,
    )
    forged_element = CharacterRingElement.model_construct(
        table=forged_table, irreducible_multiplicities=(1,)
    )
    request = CharacterTensorProductRequest.model_construct(
        left=forged_element, right=forged_element
    )

    def unexpected_group_order(*args, **kwargs):
        raise AssertionError("source-only admission must precede group_order")

    monkeypatch.setattr(
        "jacobian.math.groups.characters.representation_ring_operations.group_order",
        unexpected_group_order,
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_tensor_product(request)
