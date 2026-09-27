from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.operations import group_conjugacy_classes


def _regular_permutation(
    elements: tuple[tuple[int, int], ...],
    multiply: Callable[[tuple[int, int], tuple[int, int]], tuple[int, int]],
    left: tuple[int, int],
) -> tuple[int, ...]:
    index = {element: position for position, element in enumerate(elements)}
    return tuple(index[multiply(left, element)] for element in elements)


def _d8_multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    rotation, reflection = left
    other_rotation, other_reflection = right
    signed_rotation = -other_rotation if reflection else other_rotation
    return ((rotation + signed_rotation) % 4, (reflection + other_reflection) % 2)


def _q8_multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    """Multiply signed quaternion units (sign, basis 1/i/j/k)."""
    sign, basis = left
    other_sign, other_basis = right
    if basis == 0:
        return sign * other_sign, other_basis
    if other_basis == 0:
        return sign * other_sign, basis
    if basis == other_basis:
        return -sign * other_sign, 0
    positive = {(1, 2): 3, (2, 3): 1, (3, 1): 2}
    if (basis, other_basis) in positive:
        return sign * other_sign, positive[(basis, other_basis)]
    return -sign * other_sign, positive[(other_basis, basis)]


def _regular_group(
    kind: str,
) -> tuple[PermutationGroup, dict[tuple[int, ...], tuple[int, int]]]:
    if kind == "D8":
        elements: tuple[tuple[int, int], ...] = tuple(
            (rotation, reflection) for rotation in range(4) for reflection in range(2)
        )
        multiply = _d8_multiply
        generators = ((1, 0), (0, 1))
    else:
        elements = tuple((sign, basis) for sign in (1, -1) for basis in range(4))
        multiply = _q8_multiply
        generators = ((1, 1), (1, 2))
    generator_permutations = tuple(
        _regular_permutation(elements, multiply, generator) for generator in generators
    )
    group = PermutationGroup(degree=8, generators=generator_permutations)
    element_by_permutation = {
        _regular_permutation(elements, multiply, element): element
        for element in elements
    }
    return group, element_by_permutation


def _typed_partition(group: PermutationGroup) -> GroupConjugacyClassesResult:
    classes = group_conjugacy_classes(
        group.degree, [list(generator) for generator in group.generators]
    )
    return GroupConjugacyClassesResult(
        source=group,
        classes=tuple(
            tuple(tuple(element) for element in conjugacy_class)
            for conjugacy_class in classes
        ),
    )


def _reference_rows(
    kind: str,
    partition: GroupConjugacyClassesResult,
    element_by_permutation: dict[tuple[int, ...], tuple[int, int]],
) -> Counter[tuple[int, ...]]:
    rows = []
    if kind == "D8":
        for rotation_sign in (-1, 1):
            for reflection_sign in (-1, 1):
                rows.append(
                    tuple(
                        rotation_sign ** element_by_permutation[tuple(cls[0])][0]
                        * reflection_sign ** element_by_permutation[tuple(cls[0])][1]
                        for cls in partition.classes
                    )
                )
        # Trace of r^i s^j in the exact two-dimensional representation
        # r=[[0,-1],[1,0]], s=[[1,0],[0,-1]].
        rows.append(
            tuple(
                (
                    2
                    if element_by_permutation[tuple(cls[0])] == (0, 0)
                    else -2
                    if element_by_permutation[tuple(cls[0])] == (2, 0)
                    else 0
                )
                for cls in partition.classes
            )
        )
    else:
        for i_sign in (-1, 1):
            for j_sign in (-1, 1):
                rows.append(
                    tuple(
                        1
                        if element_by_permutation[tuple(cls[0])][1] == 0
                        else i_sign
                        if element_by_permutation[tuple(cls[0])][1] == 1
                        else j_sign
                        if element_by_permutation[tuple(cls[0])][1] == 2
                        else i_sign * j_sign
                        for cls in partition.classes
                    )
                )
        # The defining two-dimensional quaternion representation has traces
        # 2 on 1, -2 on -1, and 0 on the six noncentral units.
        rows.append(
            tuple(
                2 * element_by_permutation[tuple(cls[0])][0]
                if element_by_permutation[tuple(cls[0])][1] == 0
                else 0
                for cls in partition.classes
            )
        )
    return Counter(rows)


@pytest.mark.parametrize("kind", ("D8", "Q8"))
def test_order_eight_nonabelian_character_table_matches_direct_representations(
    kind: str,
) -> None:
    group, element_by_permutation = _regular_group(kind)
    raw_partition = _typed_partition(group)
    result = character_table(raw_partition)
    actual_rows = Counter(
        tuple(value.coefficients[0].as_fraction() for value in row.values)
        for row in result.rows
    )

    assert result.degree_square_sum == 8
    assert result.axis.cyclotomic_order == 1
    assert result.partition.source == group
    assert actual_rows == _reference_rows(
        kind, result.partition, element_by_permutation
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result

    # Independent weighted character inner products verify the explicit
    # matrices/homomorphisms form a complete orthonormal irreducible family.
    rows = tuple(actual_rows.elements())
    sizes = tuple(len(cls) for cls in result.partition.classes)
    for left in rows:
        for right in rows:
            product = (
                sum(
                    (
                        size * a * b
                        for size, a, b in zip(sizes, left, right, strict=True)
                    ),
                    Fraction(0),
                )
                / 8
            )
            assert product == int(left == right)


def test_order_eight_abelian_noncyclic_group_is_not_misclassified() -> None:
    # The regular action of C2 x C2 x C2 has eight singleton classes. It is
    # abelian but is not among the current explicit table families.
    elements = tuple((a, b, c) for a in range(2) for b in range(2) for c in range(2))

    def multiply(
        left: tuple[int, int, int], right: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        return (
            (left[0] + right[0]) % 2,
            (left[1] + right[1]) % 2,
            (left[2] + right[2]) % 2,
        )

    element_index = {element: index for index, element in enumerate(elements)}
    generators = tuple(
        tuple(element_index[multiply(generator, element)] for element in elements)
        for generator in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    )
    group = PermutationGroup(degree=8, generators=generators)
    partition = _typed_partition(group)

    with pytest.raises(OperationDomainValidationError) as exc_info:
        character_table(partition)

    assert exc_info.value.errors()[0]["type"] == "groups.characters.unsupported_group"


def test_order_eight_malformed_partition_is_rejected_before_table_construction() -> (
    None
):
    group, _ = _regular_group("D8")
    partition = _typed_partition(group)
    malformed = [list(cls) for cls in partition.classes]
    malformed[0][0] = malformed[1][0]
    malformed_partition = GroupConjugacyClassesResult.model_construct(
        source=group,
        classes=tuple(tuple(tuple(element) for element in cls) for cls in malformed),
    )

    with pytest.raises(OperationDomainValidationError):
        character_table(malformed_partition)
