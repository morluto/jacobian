"""Exact tensor products in supported finite representation rings."""

from itertools import permutations

import pytest

import jacobian.math.groups.characters.representation_ring_operations as representation_ring_operations
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterRow,
    CharacterTableResult,
    ConjugacyClassPartition,
)
from jacobian.math.groups.characters.operations import (
    character_table,
    class_function_pointwise_product,
)
from jacobian.math.groups.characters.representation_ring_operations import (
    character_tensor_product,
)
from jacobian.math.groups.operations import group_conjugacy_classes


def _s3_table() -> CharacterTableResult:
    source = PermutationGroup(degree=3, generators=((1, 2, 0), (1, 0, 2)))
    classes = group_conjugacy_classes(3, [list(g) for g in source.generators])
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in classes)
    )
    return character_table(partition)


def _element(
    table: CharacterTableResult, coordinates: tuple[int, ...]
) -> CharacterRingElement:
    return CharacterRingElement(
        table=table, irreducible_multiplicities=tuple(coordinates)
    )


def _direct_s3_tensor_square_multiplicities() -> tuple[int, ...]:
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
        _element(table, (0, 0, 1)), _element(table, (0, 0, 1))
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
        _element(table, (0, 1, 0)), _element(table, (0, 0, 1))
    )
    assert result.irreducible_multiplicities == (1, 0, 0)
    virtual = character_tensor_product(
        _element(table, (-1, 1, 0)), _element(table, (1, 0, 0))
    )
    assert virtual.irreducible_multiplicities == (-1, 1, 0)


def test_c5_with_redundant_generators_keeps_supported_cyclic_contract() -> None:
    degree = 5
    generator = tuple((point + 1) % degree for point in range(degree))
    generator_square = tuple(generator[generator[point]] for point in range(degree))
    source = PermutationGroup(degree=degree, generators=(generator, generator_square))
    classes = group_conjugacy_classes(
        degree, [list(item) for item in source.generators]
    )
    table = character_table(
        GroupConjugacyClassesResult._from_kernel(
            source, tuple(tuple(tuple(item) for item in cls) for cls in classes)
        )
    )
    # For C5, rho_k(rot)=zeta_5^k, so rho_1 tensor rho_1 = rho_2.
    product = character_tensor_product(
        _element(table, (0, 1, 0, 0, 0)), _element(table, (0, 1, 0, 0, 0))
    )
    assert product.irreducible_multiplicities == (0, 0, 1, 0, 0)


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
        character_tensor_product(_element(table, (1, 0, 0)), _element(other, (1, 0, 0)))
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
            CharacterRingElement.model_construct(
                table=forged, irreducible_multiplicities=(1, 0, 0)
            ),
            _element(table, (1, 0, 0)),
        )


def test_oversized_model_constructed_coordinates_rejected_before_group_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _s3_table()
    oversized = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(10**513, 0, 0)
    )

    def unexpected_group_expansion(*args: object, **kwargs: object) -> None:
        raise AssertionError("group order must not run before raw input admission")

    # main derives the backend group once, through _admitted_backend_group.
    monkeypatch.setattr(
        representation_ring_operations,
        "_admitted_backend_group",
        unexpected_group_expansion,
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_tensor_product(oversized, _element(table, (1, 0, 0)))


def test_huge_model_constructed_integer_never_reaches_decimal_conversion() -> None:
    table = _s3_table()
    huge = CharacterRingElement.model_construct(
        table=table, irreducible_multiplicities=(10**5000, 0, 0)
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_tensor_product(huge, _element(table, (1, 0, 0)))


def test_forged_order_one_partition_cannot_trigger_large_group_order_call(
    monkeypatch: pytest.MonkeyPatch,
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

    def unexpected_group_order(*args: object, **kwargs: object) -> None:
        raise AssertionError("source-only admission must precede the backend call")

    monkeypatch.setattr(
        representation_ring_operations,
        "_admitted_backend_group",
        unexpected_group_order,
    )
    with pytest.raises(OperationResourceAdmissionError):
        character_tensor_product(forged_element, forged_element)


def test_tensor_product_shares_the_decomposition_kernel() -> None:
    # The product is decomposed by the existing ring-decomposition kernel
    # rather than a second, parallel one, so a tensor product and a direct
    # class-function decomposition of the same class function agree exactly and
    # share one admission path.
    table = _s3_table()
    character = _element(table, (0, 0, 1))
    product = character_tensor_product(character, character)
    expanded = representation_ring_operations._expand_ring_element(character, table)
    direct = representation_ring_operations.class_function_character_decomposition(
        class_function_pointwise_product(expanded, expanded)
    )
    assert (
        product.irreducible_multiplicities
        == direct.ring_element.irreducible_multiplicities
    )
