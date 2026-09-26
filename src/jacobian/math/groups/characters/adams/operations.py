"""Bounded exact Adams operations on the finite representation ring."""

from __future__ import annotations

from fractions import Fraction

from pydantic import ValidationError

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult
from jacobian.math.groups.characters._cyclotomic import euler_phi
from jacobian.math.groups.characters._models import (
    MAX_CYCLOTOMIC_ORDER,
    CharacterRingElement,
    CharacterTableResult,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.adams._models import AdamsOperationRequest
from jacobian.math.groups.characters.operations import (
    _make_value,
    _permutation_power,
    character_table,
)
from jacobian.math.groups.characters.representation_ring_operations import (
    MAX_CHARACTER_TENSOR_PRODUCT_WORK,
    MAX_VALUE_COEFFICIENT_DIGITS,
    _admit_ring_element_shape,
    _admit_source_group_order,
    _coordinates_on_authenticated_table,
)
from jacobian.math.groups.operations import group_conjugacy_classes


def _invalid(
    code: str, message: str, location: tuple[str, ...]
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _admit_adams_work(
    element: CharacterRingElement,
    exponent: int,
    source_work: int,
    group_order: int,
) -> None:
    table = element.table
    classes = len(table.partition.classes)
    degree = table.partition.source.degree
    rows = len(table.rows)
    cyclotomic_dimension = euler_phi(table.axis.cyclotomic_order)
    exponent_bits = exponent.bit_length()
    # Bound representative powering, class membership indexing, virtual-value
    # expansion, and the exact inner products which recover irreducible rows.
    power_map_work = classes * degree * (3 * exponent_bits + 2)
    membership_work = group_order * degree
    expansion_work = classes * rows * cyclotomic_dimension
    pairing_work = rows * classes * cyclotomic_dimension**2
    additional_work = power_map_work + membership_work + expansion_work + pairing_work
    coefficient_digits = max(
        1, *(len(str(abs(value))) for value in element.irreducible_multiplicities)
    )
    value_digits = max(
        1,
        *(
            max(len(str(abs(value.num))), len(str(value.den)))
            for row in table.rows
            for class_value in row.values
            for value in class_value.coefficients
        ),
    )
    predicted_digits = coefficient_digits + value_digits + len(str(rows))
    work = (
        additional_work
        + source_work
        + 2 * rows * classes * cyclotomic_dimension * predicted_digits**2
    )
    if work > MAX_CHARACTER_TENSOR_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="groups.characters.adams_work_exceeds_envelope",
            message="Adams class expansion and exact pairings exceed the work envelope",
        )
    if predicted_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="groups.characters.adams_output_height_exceeds_envelope",
            message="Adams coordinates exceed the exact coefficient envelope",
        )


def character_adams_operation(request: AdamsOperationRequest) -> CharacterRingElement:
    """Return the virtual character whose value at g is chi(g**k).

    The table remains bound to its concrete group. The class-power map is
    computed from its canonical complete conjugacy partition, and the result
    is expressed in that table's irreducible basis.
    """
    if not isinstance(request, AdamsOperationRequest):
        raise _invalid(
            "groups.characters.adams_request_type",
            "request must contain a virtual character and positive exponent",
            ("request",),
        )
    try:
        request = AdamsOperationRequest.model_validate(request.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _invalid(
            "groups.characters.adams_invalid_request",
            "request has a malformed character or exponent",
            ("request",),
        ) from exc

    element = request.character
    _admit_ring_element_shape(element, "character")
    source = element.table.partition.source
    group_order, source_work = _admit_source_group_order(source)
    if group_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("character", "table", "partition", "source"),
            code="groups.characters.adams_group_order_exceeds_envelope",
            message="Adams operations currently admit group order at most 60",
        )
    _admit_adams_work(element, request.exponent, source_work, group_order)

    raw_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    raw_partition = GroupConjugacyClassesResult._from_kernel(
        source,
        tuple(
            tuple(tuple(group_element) for group_element in cls) for cls in raw_classes
        ),
    )
    table: CharacterTableResult = character_table(raw_partition)
    if element.table != table:
        raise _invalid(
            "groups.characters.adams_noncanonical_table",
            "input must retain the exact canonical character table for its group",
            ("character", "table"),
        )

    element_to_class = {
        tuple(group_element): class_index
        for class_index, conjugacy_class in enumerate(table.partition.classes)
        for group_element in conjugacy_class
    }
    class_power_images: list[int] = []
    for conjugacy_class in table.partition.classes:
        representative = tuple(conjugacy_class[0])
        powered = _permutation_power(representative, request.exponent)
        try:
            class_power_images.append(element_to_class[powered])
        except KeyError as exc:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc

    order = table.axis.cyclotomic_order
    dimension = euler_phi(order)
    values: list[CyclotomicValue] = []
    for image_index in class_power_images:
        coefficients = [Fraction(0) for _ in range(dimension)]
        for multiplicity, row in zip(
            element.irreducible_multiplicities, table.rows, strict=True
        ):
            for power, coefficient in enumerate(row.values[image_index].coefficients):
                coefficients[power] += multiplicity * coefficient.as_fraction()
        values.append(_make_value(order, tuple(coefficients)))
    adams_class_function = FiniteClassFunction._from_kernel(
        axis=table.axis,
        values=tuple(values),
    )
    return _coordinates_on_authenticated_table(adams_class_function, table)


__all__ = ["character_adams_operation"]
