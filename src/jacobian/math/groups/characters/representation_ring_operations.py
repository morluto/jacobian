"""Coordinates of class functions in bounded finite-group character bases."""

from __future__ import annotations

from fractions import Fraction

from pydantic import ValidationError

from jacobian._exact import canonical_rational_component_digits
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import GroupConjugacyClassesResult, PermutationGroup
from jacobian.math.groups.characters._cyclotomic import (
    add_values,
    conjugate_value,
    euler_phi,
    multiply_values,
    scale_value,
    value_from_power,
    zero_value,
)
from jacobian.math.groups.characters._models import (
    MAX_CHARACTER_TABLE_CELLS,
    MAX_CLASS_COUNT,
    MAX_CYCLOTOMIC_ORDER,
    MAX_GROUP_ORDER,
    MAX_VALUE_COEFFICIENT_DIGITS,
    CharacterRingDecompositionRequest,
    CharacterRingDecompositionResult,
    CharacterRingElement,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import (
    _admit_inner_product,
    _fractions,
    _make_value,
    character_table,
    _character_table_from_admitted_partition,
)
from jacobian.math.groups.operations import group_conjugacy_classes, group_order

MAX_CHARACTER_RING_DECOMPOSITION_WORK = 50_000_000
MAX_CHARACTER_RING_DECOMPOSITION_OUTPUT_BYTES = 10_000_000


def _invalid(
    code: str, message: str, location: tuple[str, ...]
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _admit_output(
    *,
    source: PermutationGroup,
    source_function: FiniteClassFunction,
    concrete_order: int,
) -> None:
    """Bound a worst-case complete table and coordinate vector serialization."""
    dimension = euler_phi(concrete_order)
    table_cells = concrete_order * concrete_order * dimension
    if table_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("class_function",),
            code="groups.characters.ring_decomposition_table_exceeds_envelope",
            message="canonical character table exceeds the exact cell envelope",
        )
    group = source
    group_bytes = 512 + len(group.generators) * (group.degree * 8 + 16)
    input_class_count = len(source_function.axis.class_sizes)
    source_digits = max(
        canonical_rational_component_digits(coefficient)
        for value in source_function.values
        for coefficient in value.coefficients
    )
    source_function_bytes = (
        input_class_count
        * euler_phi(source_function.axis.cyclotomic_order)
        * (2 * source_digits + 32)
    )
    output_bytes = (
        table_cells * 40
        + concrete_order * (MAX_VALUE_COEFFICIENT_DIGITS * 2 + 32)
        + source_function_bytes
        + group_bytes * 3
        + concrete_order * group.degree * 12
        + 2 * concrete_order * group.degree * 12
        + 65_536
    )
    if (
        output_bytes > MAX_CHARACTER_RING_DECOMPOSITION_OUTPUT_BYTES
        or output_bytes > CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("class_function",),
            code="groups.characters.ring_decomposition_output_exceeds_envelope",
            message="character-ring decomposition exceeds its exact output envelope",
        )


def _admit_pairings_before_expansion(
    function: FiniteClassFunction, concrete_order: int
) -> None:
    """Admit a worst-case canonical row before enumerating group classes."""
    order = concrete_order
    dimension = euler_phi(order)
    table_order_roots = tuple(
        value_from_power(order, exponent) for exponent in range(order)
    )
    root_coefficient_bound = max(
        abs(coefficient.numerator)
        for value in table_order_roots
        for coefficient in value
    )
    # The only supported noncyclic table is S3, whose largest entry is 2.
    row_coefficient_bound = max(root_coefficient_bound, 2 if order == 6 else 1)
    synthetic_row_value = _make_value(
        order, (Fraction(row_coefficient_bound),) * dimension
    )
    target_axis = function.axis.model_copy(update={"cyclotomic_order": order})
    lifted = function
    if function.axis.cyclotomic_order == 1 and order != 1:
        lifted = FiniteClassFunction._from_kernel(
            axis=target_axis,
            values=tuple(
                _make_value(
                    order,
                    (_fractions(value)[0],) + (Fraction(0),) * (dimension - 1),
                )
                for value in function.values
            ),
        )
    synthetic_row = FiniteClassFunction._from_kernel(
        axis=target_axis,
        values=(synthetic_row_value,) * len(function.axis.class_sizes),
    )
    _admit_inner_product(lifted, synthetic_row)

    input_digits = max(
        canonical_rational_component_digits(coefficient)
        for value in lifted.values
        for coefficient in value.coefficients
    )
    row_digits = len(str(row_coefficient_bound))
    pair_digits = max(input_digits, row_digits)
    # One pairing visits every class. Multiplication uses dim^2 exact
    # coefficient products, reduction is O(dim^2), and conjugation is O(order).
    # There are at most |G| irreducible rows and at most |G| classes.
    aggregate_work = (
        len(function.axis.class_sizes)
        * order
        * (order + 4 * dimension * dimension)
        * pair_digits
        * pair_digits
    )
    if aggregate_work > MAX_CHARACTER_RING_DECOMPOSITION_WORK:
        raise OperationResourceAdmissionError(
            location=("class_function",),
            code="groups.characters.ring_decomposition_work_exceeds_envelope",
            message="all irreducible pairings exceed the decomposition work envelope",
        )


def _admit_before_class_expansion(
    function: FiniteClassFunction,
    source: PermutationGroup,
    concrete_order: int,
) -> None:
    """Admit input arithmetic and worst-case output before conjugacy expansion."""
    if (
        function.axis.group_order != concrete_order
        or len(function.axis.class_sizes) > concrete_order
        or function.axis.cyclotomic_order not in (1, concrete_order)
    ):
        raise _invalid(
            "groups.characters.ring_decomposition_axis_mismatch",
            "class function must use a possible class axis and rational or table coefficient field",
            ("class_function", "axis"),
        )
    _admit_pairings_before_expansion(function, concrete_order)
    _admit_output(
        source=source,
        source_function=function,
        concrete_order=concrete_order,
    )


def _inner_product_value(
    function: FiniteClassFunction,
    row: tuple[CyclotomicValue, ...],
    *,
    order: int,
) -> CyclotomicValue:
    total = zero_value(order)
    for class_size, value, character_value in zip(
        function.axis.class_sizes, function.values, row, strict=True
    ):
        conjugate = conjugate_value(order, _fractions(character_value))
        product = multiply_values(order, _fractions(value), conjugate)
        total = add_values(
            order, total, scale_value(order, Fraction(class_size), product)
        )
    return _make_value(
        order,
        tuple(coefficient / function.axis.group_order for coefficient in total),
    )


def class_function_character_decomposition(
    request: CharacterRingDecompositionRequest,
) -> CharacterRingDecompositionResult:
    """Return exact irreducible coordinates when a bounded class function is virtual.

    The table is regenerated from the function's concrete group. The current
    canonical character-table implementation supports the trivial, cyclic, and
    S3 permutation groups. A class function has integer irreducible
    coordinates exactly when it is a virtual character in this complete
    orthonormal basis.
    """
    if not isinstance(request, CharacterRingDecompositionRequest):
        raise _invalid(
            "groups.characters.ring_decomposition_request_type",
            "request must be a character-ring decomposition request",
            ("request",),
        )
    try:
        request = CharacterRingDecompositionRequest.model_validate(request.model_dump())
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _invalid(
            "groups.characters.invalid_ring_decomposition_request",
            "request has a malformed class function",
            ("request",),
        ) from exc

    function = request.class_function
    source = function.axis.group
    if source is None:
        raise _invalid(
            "groups.characters.ring_decomposition_requires_group",
            "decomposition requires a class axis bound to a concrete group",
            ("class_function", "axis"),
        )
    if (
        function.axis.group_order > MAX_GROUP_ORDER
        or len(function.axis.class_sizes) > MAX_CLASS_COUNT
        or function.axis.cyclotomic_order > MAX_CYCLOTOMIC_ORDER
    ):
        raise OperationResourceAdmissionError(
            location=("class_function", "axis"),
            code="groups.characters.ring_decomposition_axis_exceeds_envelope",
            message="class-function axis exceeds the bounded character envelope",
        )

    # The canonical table operation has a strict order envelope. Check the
    # generated group before materializing its complete class partition.
    concrete_order = group_order(source)
    if concrete_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("class_function", "axis", "group"),
            code="groups.characters.ring_decomposition_group_order_exceeds_envelope",
            message=(
                "character decomposition currently admits concrete group order "
                f"at most {MAX_CYCLOTOMIC_ORDER}"
            ),
        )
    _admit_before_class_expansion(function, source, concrete_order)
    raw_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    partition = GroupConjugacyClassesResult._from_kernel(
        source,
        tuple(tuple(tuple(element) for element in cls) for cls in raw_classes),
    )
    table = _character_table_from_admitted_partition(partition)
    input_axis = function.axis
    table_axis = table.axis
    same_class_axis = (
        input_axis.class_sizes == table_axis.class_sizes
        and input_axis.group_order == table_axis.group_order
        and input_axis.group == table_axis.group
        and input_axis.class_representatives == table_axis.class_representatives
    )
    if not same_class_axis or input_axis.cyclotomic_order not in (
        1,
        table_axis.cyclotomic_order,
    ):
        raise _invalid(
            "groups.characters.ring_decomposition_axis_mismatch",
            "class function must use the canonical class axis and either the table field or its rational subfield",
            ("class_function", "axis"),
        )

    source_class_function = function
    # Rational values have a unique exact inclusion into every cyclotomic
    # table field. No nontrivial field identification is inferred here.
    if input_axis.cyclotomic_order == 1 and table_axis.cyclotomic_order != 1:
        promoted_values = tuple(
            _make_value(
                table_axis.cyclotomic_order,
                (_fractions(value)[0],)
                + (Fraction(0),) * (euler_phi(table_axis.cyclotomic_order) - 1),
            )
            for value in function.values
        )
        function = FiniteClassFunction._from_kernel(
            axis=table_axis, values=promoted_values
        )

    order = table.axis.cyclotomic_order
    if order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("class_function", "axis", "cyclotomic_order"),
            code="groups.characters.ring_decomposition_cyclotomic_order_exceeds_envelope",
            message="character decomposition exceeds the exact cyclotomic envelope",
        )

    # All arithmetic, table, and result bounds have been admitted before class
    # expansion. In the complete orthonormal basis, these Hermitian pairings
    # are the unique irreducible coordinates.
    coordinates: list[int] = []
    for row in table.rows:
        inner = _inner_product_value(function, row.values, order=order)
        if inner.coefficients[0].den != 1 or any(
            coefficient.num != 0 for coefficient in inner.coefficients[1:]
        ):
            raise _invalid(
                "groups.characters.not_virtual_character",
                "class function has a nonintegral irreducible coordinate",
                ("class_function",),
            )
        coordinate = inner.coefficients[0].num
        if coordinate.bit_length() > 1702:
            raise OperationResourceAdmissionError(
                location=("class_function",),
                code="groups.characters.ring_coordinate_exceeds_envelope",
                message="irreducible coordinate exceeds the exact integer envelope",
            )
        coordinates.append(coordinate)

    element = CharacterRingElement._from_kernel(
        table=table, irreducible_multiplicities=tuple(coordinates)
    )
    return CharacterRingDecompositionResult._from_kernel(
        source_class_function=source_class_function, ring_element=element
    )


__all__ = [
    "MAX_CHARACTER_RING_DECOMPOSITION_OUTPUT_BYTES",
    "MAX_CHARACTER_RING_DECOMPOSITION_WORK",
    "class_function_character_decomposition",
]
