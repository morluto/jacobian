"""Exact class-function coordinates in the finite representation ring."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

import jacobian.math.groups.characters.representation_ring_operations as ring_operations
from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._cyclotomic import euler_phi
from jacobian.math.groups.characters._models import (
    CharacterRingDecompositionRequest,
    CharacterRingElement,
    ClassAxis,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.characters.representation_ring_operations import (
    class_function_character_decomposition,
)
from jacobian.math.groups.operations import group_conjugacy_classes


def _partition(generators: tuple[tuple[int, ...], ...]) -> GroupConjugacyClassesResult:
    degree = len(generators[0])
    source = PermutationGroup(degree=degree, generators=generators)
    classes = group_conjugacy_classes(
        degree, [list(generator) for generator in generators]
    )
    return GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(element) for element in row) for row in classes)
    )


def _function_from_rows(table, coordinates: tuple[int, ...]) -> FiniteClassFunction:
    order = table.axis.cyclotomic_order
    values: list[CyclotomicValue] = []
    for class_index in range(len(table.axis.class_sizes)):
        coefficients = [
            Fraction(0)
            for _ in range(len(table.rows[0].values[class_index].coefficients))
        ]
        for coefficient, row in zip(coordinates, table.rows, strict=True):
            for index, value in enumerate(row.values[class_index].coefficients):
                coefficients[index] += coefficient * value.as_fraction()
        values.append(
            CyclotomicValue(
                order=order,
                coefficients=tuple(
                    CanonicalRational(num=value.numerator, den=value.denominator)
                    for value in coefficients
                ),
            )
        )
    return FiniteClassFunction(axis=table.axis, values=tuple(values))


def _s3_class_function(
    values: tuple[Fraction, Fraction, Fraction],
) -> FiniteClassFunction:
    partition = _partition(((1, 2, 0), (1, 0, 2)))
    axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(row) for row in partition.classes),
        cyclotomic_order=1,
        group=partition.source,
        class_representatives=tuple(row[0] for row in partition.classes),
    )
    return FiniteClassFunction(
        axis=axis,
        values=tuple(
            CyclotomicValue(
                order=1,
                coefficients=(CanonicalRational.from_fraction(value),),
            )
            for value in values
        ),
    )


def _function_on_partition(
    partition: GroupConjugacyClassesResult,
    value: Fraction,
    *,
    cyclotomic_order: int = 1,
) -> FiniteClassFunction:
    axis = ClassAxis._from_kernel(
        class_sizes=tuple(len(row) for row in partition.classes),
        cyclotomic_order=cyclotomic_order,
        group=partition.source,
        class_representatives=tuple(row[0] for row in partition.classes),
    )
    dimension = euler_phi(cyclotomic_order)
    values = tuple(
        CyclotomicValue(
            order=cyclotomic_order,
            coefficients=(CanonicalRational.from_fraction(value),)
            + (CanonicalRational(num=0, den=1),) * (dimension - 1),
        )
        for _ in partition.classes
    )
    return FiniteClassFunction(axis=axis, values=values)


def test_s3_irreducible_and_reducible_virtual_coordinates() -> None:
    standard = _s3_class_function((Fraction(2), Fraction(0), Fraction(-1)))
    result = class_function_character_decomposition(
        CharacterRingDecompositionRequest(class_function=standard)
    )
    assert result.ring_element.irreducible_multiplicities == (0, 0, 1)
    assert (
        CharacterRingElement.model_validate_json(result.ring_element.model_dump_json())
        == result.ring_element
    )

    reducible = _s3_class_function((Fraction(3), Fraction(1), Fraction(0)))
    decomposed = class_function_character_decomposition(
        CharacterRingDecompositionRequest(class_function=reducible)
    )
    assert decomposed.ring_element.irreducible_multiplicities == (1, 0, 1)


def test_s3_virtual_character_keeps_signed_coordinates() -> None:
    virtual = _s3_class_function((Fraction(-1), Fraction(1), Fraction(2)))
    result = class_function_character_decomposition(
        CharacterRingDecompositionRequest(class_function=virtual)
    )
    assert result.ring_element.irreducible_multiplicities == (1, 0, -1)


def test_cyclic_complex_character_uses_exact_hermitian_pairing() -> None:
    table = character_table(_partition(((1, 2, 0),)))
    for index in range(len(table.rows)):
        function = _function_from_rows(
            table,
            tuple(int(row_index == index) for row_index in range(len(table.rows))),
        )
        result = class_function_character_decomposition(
            CharacterRingDecompositionRequest(class_function=function)
        )
        assert result.ring_element.irreducible_multiplicities == tuple(
            int(row_index == index) for row_index in range(len(table.rows))
        )


def test_nonintegral_class_function_is_not_a_virtual_character() -> None:
    half_trivial = _s3_class_function((Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)))
    with pytest.raises(OperationDomainValidationError):
        class_function_character_decomposition(
            CharacterRingDecompositionRequest(class_function=half_trivial)
        )


def test_work_and_height_are_admitted_before_conjugacy_expansion(monkeypatch) -> None:
    partition = _partition(((1, 2, 0),))
    oversized = _function_on_partition(partition, Fraction(10**511))

    def unexpected_expansion(*args, **kwargs):
        pytest.fail("conjugacy classes expanded before exact arithmetic admission")

    monkeypatch.setattr(
        ring_operations, "group_conjugacy_classes", unexpected_expansion
    )
    with pytest.raises(OperationResourceAdmissionError, match=r"height|envelope"):
        class_function_character_decomposition(
            CharacterRingDecompositionRequest(class_function=oversized)
        )


def test_maximum_supported_cyclic_group_decomposes_exactly() -> None:
    partition = _partition(((*range(1, 60), 0),))
    trivial = _function_on_partition(partition, Fraction(1))
    result = class_function_character_decomposition(
        CharacterRingDecompositionRequest(class_function=trivial)
    )
    assert result.ring_element.irreducible_multiplicities == (1,) + (0,) * 59


def test_trivial_group_decomposition_and_round_trip() -> None:
    degree_zero_group = PermutationGroup(degree=0, generators=((),))
    degree_zero_classes = group_conjugacy_classes(0, [[]])
    degree_zero_partition = GroupConjugacyClassesResult._from_kernel(
        degree_zero_group,
        tuple(tuple(tuple(element) for element in cls) for cls in degree_zero_classes),
    )
    degree_zero_function = _function_on_partition(degree_zero_partition, Fraction(7))
    degree_zero_result = class_function_character_decomposition(
        CharacterRingDecompositionRequest(class_function=degree_zero_function)
    )
    assert degree_zero_result.ring_element.irreducible_multiplicities == (7,)
    assert type(degree_zero_result).model_validate_json(
        degree_zero_result.model_dump_json()
    ) == degree_zero_result

    partition = _partition(((0,),))
    constant = _function_on_partition(partition, Fraction(7))
    result = class_function_character_decomposition(
        CharacterRingDecompositionRequest(class_function=constant)
    )
    assert result.ring_element.irreducible_multiplicities == (7,)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_group_order_above_table_envelope_rejects_before_class_expansion(
    monkeypatch,
) -> None:
    generator = (*range(1, 61), 0)
    source = PermutationGroup(degree=61, generators=(generator,))
    axis = ClassAxis._from_kernel(
        class_sizes=(1,) * 61,
        cyclotomic_order=1,
        group=source,
        class_representatives=(tuple(range(61)),) * 61,
    )
    one = CyclotomicValue(order=1, coefficients=(CanonicalRational(num=1, den=1),))
    function = FiniteClassFunction(axis=axis, values=(one,) * 61)

    def unexpected_expansion(*args, **kwargs):
        pytest.fail("group-order rejection must precede conjugacy expansion")

    monkeypatch.setattr(
        ring_operations, "group_conjugacy_classes", unexpected_expansion
    )
    with pytest.raises(OperationResourceAdmissionError, match="group order"):
        class_function_character_decomposition(
            CharacterRingDecompositionRequest(class_function=function)
        )


def test_ring_element_coordinates_are_strict_integers() -> None:
    table = character_table(_partition(((1, 2, 0), (1, 0, 2))))
    with pytest.raises(ValidationError):
        CharacterRingElement.model_validate(
            {"table": table.model_dump(), "irreducible_multiplicities": [True, 0, 0]}
        )


def test_catalog_example_executes() -> None:
    catalog = Catalog.open()
    operation_id = "finite_group.class_function.character_ring_decompose.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None
    result = invoke_operation(operation_id, operation.examples[0].input, catalog)
    assert result.output["ring_element"]["irreducible_multiplicities"] == [0, 0, 1]
