"""Coordinates of class functions in bounded finite-group character bases."""

from __future__ import annotations

import math
from fractions import Fraction

from pydantic import ValidationError

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.canonical import CanonicalLimits
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import (
    MAX_GROUP_DEGREE,
    GroupConjugacyClassesResult,
    PermutationGroup,
)
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
    CharacterCenter,
    CharacterCenterRequest,
    CharacterExteriorSquareRequest,
    CharacterKernel,
    CharacterKernelRequest,
    CharacterRingDecompositionRequest,
    CharacterRingDecompositionResult,
    CharacterRingElement,
    CharacterRow,
    CharacterSymmetricSquareRequest,
    CharacterTableResult,
    CharacterTensorProductRequest,
    ConjugacyClassPartition,
    CyclotomicValue,
    FiniteClassFunction,
)
from jacobian.math.groups.characters.operations import (
    _admit_inner_product,
    _fractions,
    _make_value,
    character_table,
    class_function_pointwise_product,
)
from jacobian.math.groups.operations import group_conjugacy_classes, group_order

MAX_CHARACTER_RING_DECOMPOSITION_WORK = 50_000_000
MAX_CHARACTER_RING_DECOMPOSITION_OUTPUT_BYTES = 10_000_000
MAX_CHARACTER_TENSOR_PRODUCT_WORK = 50_000_000
MAX_CHARACTER_TENSOR_PRODUCT_OUTPUT_BYTES = 10_000_000
MAX_CHARACTER_KERNEL_WORK = 50_000_000
MAX_CHARACTER_KERNEL_OUTPUT_BYTES = 1_000_000
MAX_CHARACTER_CENTER_WORK = 50_000_000
MAX_CHARACTER_CENTER_OUTPUT_BYTES = 1_000_000


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


def _coordinates_on_authenticated_table(
    function: FiniteClassFunction, table: CharacterTableResult
) -> CharacterRingElement:
    """Compute exact irreducible coordinates using an already canonical table."""
    coordinates = []
    for row in table.rows:
        inner = _inner_product_value(
            function, row.values, order=table.axis.cyclotomic_order
        )
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
    return CharacterRingElement._from_kernel(
        table=table, irreducible_multiplicities=tuple(coordinates)
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
    table = character_table(partition)
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


def _admit_tensor_partition_shape(
    partition: ConjugacyClassPartition, name: str
) -> PermutationGroup:
    if not isinstance(partition, ConjugacyClassPartition):
        raise _invalid(
            "groups.characters.tensor_product_partition",
            "table must retain a conjugacy partition",
            (name, "table", "partition"),
        )
    source = getattr(partition, "source", None)
    if not isinstance(source, PermutationGroup):
        raise _invalid(
            "groups.characters.tensor_product_group",
            "table must retain a concrete permutation group",
            (name, "table", "partition", "source"),
        )
    degree = getattr(source, "degree", None)
    if type(degree) is not int or not 1 <= degree <= MAX_GROUP_DEGREE:
        raise OperationResourceAdmissionError(
            location=(name, "table", "partition", "source"),
            code="groups.characters.tensor_product_degree_exceeds_envelope",
            message="permutation degree exceeds the tensor-product envelope",
        )
    generators = getattr(source, "generators", None)
    if not isinstance(generators, tuple) or len(generators) > MAX_GROUP_DEGREE:
        raise OperationResourceAdmissionError(
            location=(name, "table", "partition", "source", "generators"),
            code="groups.characters.tensor_product_generators_exceed_envelope",
            message="generator count exceeds the tensor-product envelope",
        )
    for generator in generators:
        if (
            not isinstance(generator, tuple)
            or len(generator) != degree
            or any(type(x) is not int or x < 0 or x >= degree for x in generator)
            or len(set(generator)) != degree
        ):
            raise _invalid(
                "groups.characters.tensor_product_generator_shape",
                "group generators must be bounded permutations",
                (name, "table", "partition", "source", "generators"),
            )
    classes = getattr(partition, "classes", None)
    if not isinstance(classes, tuple) or not 1 <= len(classes) <= MAX_CLASS_COUNT:
        raise OperationResourceAdmissionError(
            location=(name, "table", "partition", "classes"),
            code="groups.characters.tensor_product_class_count_exceeds_envelope",
            message="class count exceeds the tensor-product envelope",
        )
    represented_elements = 0
    for cls in classes:
        if not isinstance(cls, tuple) or not cls:
            raise _invalid(
                "groups.characters.tensor_product_partition_shape",
                "partition element shapes are invalid",
                (name, "table", "partition", "classes"),
            )
        represented_elements += len(cls)
        if represented_elements > MAX_CYCLOTOMIC_ORDER:
            break
        for element_value in cls:
            if (
                not isinstance(element_value, tuple)
                or len(element_value) != degree
                or any(
                    type(x) is not int or x < 0 or x >= degree for x in element_value
                )
                or len(set(element_value)) != degree
            ):
                raise _invalid(
                    "groups.characters.tensor_product_partition_shape",
                    "partition elements must be bounded permutations",
                    (name, "table", "partition", "classes"),
                )
    if represented_elements > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=(name, "table", "partition", "classes"),
            code="groups.characters.tensor_product_group_order_exceeds_envelope",
            message="tensor products currently admit group order at most 60",
        )
    return source


def _admit_tensor_table_rows(table: CharacterTableResult, name: str) -> None:
    partition = getattr(table, "partition", None)
    if not isinstance(partition, ConjugacyClassPartition):
        raise _invalid(
            "groups.characters.tensor_product_partition",
            "table must retain a conjugacy partition",
            (name, "table", "partition"),
        )
    classes = getattr(partition, "classes", None)
    rows = getattr(table, "rows", None)
    if not isinstance(classes, tuple):
        raise _invalid(
            "groups.characters.tensor_product_partition",
            "table classes must be a bounded tuple",
            (name, "table", "partition", "classes"),
        )
    if (
        not isinstance(rows, tuple)
        or len(rows) != len(classes)
        or not 1 <= len(rows) <= MAX_CLASS_COUNT
    ):
        raise _invalid(
            "groups.characters.tensor_product_row_shape",
            "character rows exceed the bounded table envelope",
            (name, "table", "rows"),
        )
    cells = 0
    for row in rows:
        row_values = getattr(row, "values", None)
        if (
            not isinstance(row, CharacterRow)
            or not isinstance(row_values, tuple)
            or len(row_values) != len(classes)
        ):
            raise _invalid(
                "groups.characters.tensor_product_row_shape",
                "every row must cover the represented classes",
                (name, "table", "rows"),
            )
        cells += len(row_values)
        for value in row_values:
            coefficients = getattr(value, "coefficients", None)
            if (
                not isinstance(value, CyclotomicValue)
                or not isinstance(coefficients, tuple)
                or len(coefficients) > MAX_CYCLOTOMIC_ORDER
            ):
                raise _invalid(
                    "groups.characters.tensor_product_value_shape",
                    "table values must use bounded exact cyclotomic coordinates",
                    (name, "table", "rows"),
                )
            for coefficient in coefficients:
                if (
                    not isinstance(coefficient, CanonicalRational)
                    or type(coefficient.num) is not int
                    or type(coefficient.den) is not int
                    or coefficient.den <= 0
                    or coefficient.num.bit_length() > 1702
                    or coefficient.den.bit_length() > 1702
                    or max(len(str(abs(coefficient.num))), len(str(coefficient.den)))
                    > MAX_VALUE_COEFFICIENT_DIGITS
                ):
                    raise OperationResourceAdmissionError(
                        location=(name, "table", "rows"),
                        code="groups.characters.tensor_product_table_height",
                        message="table coefficient exceeds the exact coefficient envelope",
                    )
    if cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=(name, "table"),
            code="groups.characters.tensor_product_table_exceeds_envelope",
            message="character-table cells exceed the tensor-product envelope",
        )


def _admit_ring_element_shape(element: CharacterRingElement, name: str) -> None:
    """Bound untrusted model-constructed table and coordinate shapes cheaply."""
    table = getattr(element, "table", None)
    if not isinstance(table, CharacterTableResult):
        raise _invalid(
            "groups.characters.tensor_product_table",
            "input must retain a character table",
            (name, "table"),
        )
    _admit_tensor_partition_shape(table.partition, name)
    _admit_tensor_table_rows(table, name)
    coords = getattr(element, "irreducible_multiplicities", None)
    if not isinstance(coords, tuple) or len(coords) != len(table.rows):
        raise _invalid(
            "groups.characters.tensor_product_coordinate_shape",
            "one bounded coordinate is required per table row",
            (name, "irreducible_multiplicities"),
        )
    if any(
        type(value) is not int
        or value.bit_length() > 1702
        or len(str(abs(value))) > MAX_VALUE_COEFFICIENT_DIGITS
        for value in coords
    ):
        raise OperationResourceAdmissionError(
            location=(name, "irreducible_multiplicities"),
            code="groups.characters.tensor_product_coordinate_height",
            message="virtual-character coordinates exceed the exact coefficient envelope",
        )


def _admit_source_group_order(
    source: PermutationGroup,
) -> tuple[int, int]:
    """Compute source order by a source-only bounded permutation enumeration.

    Fixed points are removed from the probe's domain. A single permutation
    generates a cyclic group whose order is computed directly from its cycle
    lengths. With multiple generators, enumerate the generated group closure
    only until it has 61 elements. This accepts every represented group of
    order at most 60, regardless of permutation support, while rejecting
    larger groups after a fixed amount of source-only work.
    """
    degree = source.degree
    generators = source.generators
    work = degree * max(1, len(generators))
    support = tuple(
        point
        for point in range(degree)
        if any(generator[point] != point for generator in generators)
    )
    if not support:
        return 1, work
    position = {point: index for index, point in enumerate(support)}
    compressed = tuple(
        sorted(
            {
                tuple(position[generator[point]] for point in support)
                for generator in generators
            }
        )
    )
    active_degree = len(support)
    work += active_degree * len(generators)
    compression_bound = len(generators) ** 2 * active_degree
    work += compression_bound
    if len(compressed) == 1:
        permutation = compressed[0]
        visited: set[int] = set()
        order = 1
        for point in range(active_degree):
            if point in visited:
                continue
            current = point
            cycle_length = 0
            while current not in visited:
                visited.add(current)
                current = permutation[current]
                cycle_length += 1
            order = math.lcm(order, cycle_length)
        return order, work + active_degree
    closure_bound = (
        (MAX_CYCLOTOMIC_ORDER + 1)
        * len(compressed)
        * active_degree
        * (MAX_CYCLOTOMIC_ORDER + 1)
    )
    if work + closure_bound > MAX_CHARACTER_TENSOR_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("left", "table", "partition", "source"),
            code="groups.characters.tensor_product_group_order_work_exceeds_envelope",
            message=(
                "source-only group closure exceeds the tensor-product work envelope"
            ),
        )
    identity = tuple(range(active_degree))
    known = {identity}
    pending = [identity]
    while pending:
        current_permutation = pending.pop()
        for generator in compressed:
            candidate = tuple(
                generator[current_permutation[index]] for index in range(active_degree)
            )
            work += active_degree * (len(known) + 1) + 1
            if candidate not in known:
                known.add(candidate)
                if len(known) > MAX_CYCLOTOMIC_ORDER:
                    raise OperationResourceAdmissionError(
                        location=("left", "table", "partition", "source"),
                        code="groups.characters.tensor_product_group_order_exceeds_envelope",
                        message="tensor products currently admit group order at most 60",
                    )
                pending.append(candidate)
    return len(known), work


def _admit_tensor_arithmetic(
    left: CharacterRingElement,
    right: CharacterRingElement,
    source: PermutationGroup,
    source_work: int,
    concrete_order: int,
    additional_work: int = 0,
) -> tuple[int, int, int]:
    """Admit table expansion, product, pairings, reconstruction, and output."""
    # Before authenticating the table, its claimed class count and order may
    # understate the real group. Use the exact source order as a worst case.
    order = concrete_order
    classes = concrete_order
    rows = concrete_order
    dimension = euler_phi(order)
    coefficient_digits = max(
        1,
        *(
            len(str(abs(coefficient)))
            for element in (left, right)
            for coefficient in element.irreducible_multiplicities
        ),
    )
    table_digits = max(
        1,
        *(
            max(len(str(abs(value.num))), len(str(value.den)))
            for element in (left, right)
            for row in element.table.rows
            for class_value in row.values
            for value in class_value.coefficients
        ),
    )
    expanded_digits = coefficient_digits + table_digits + len(str(rows))
    product_digits = (
        2 * expanded_digits
        + (len(str(dimension - 1)) if dimension > 1 else 0)
        + max(0, dimension - 1)
        + 2
    )
    predicted_digits = product_digits + table_digits + len(str(order)) + (order - 1)
    # Two input expansions and one exact reconstruction of the result are
    # mandatory. This includes the source-only group-order enumeration once.
    work = (
        3 * rows * classes * dimension * expanded_digits**2
        + classes * dimension * dimension * product_digits**2
        + rows
        * classes
        * order
        * (order + 4 * dimension * dimension)
        * (product_digits + table_digits) ** 2
        + rows * classes * dimension * predicted_digits**2
        + order * order * source.degree * max(1, len(source.generators))
        + order * order * dimension * table_digits**2
        + rows * classes * dimension * table_digits**2
        + source_work
        + additional_work
    )
    if work > MAX_CHARACTER_TENSOR_PRODUCT_WORK:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="groups.characters.tensor_product_work_exceeds_envelope",
            message="table reconstruction and exact product exceed the tensor-product work envelope",
        )
    if predicted_digits > MAX_VALUE_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="groups.characters.tensor_product_output_height",
            message="predicted tensor-product coordinates exceed the exact coefficient envelope",
        )
    output_bytes = (
        rows * (2 * MAX_VALUE_COEFFICIENT_DIGITS + 32)
        + MAX_CHARACTER_TABLE_CELLS * 40
        + 65_536
    )
    if (
        output_bytes > MAX_CHARACTER_TENSOR_PRODUCT_OUTPUT_BYTES
        or output_bytes > CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="groups.characters.tensor_product_output_exceeds_envelope",
            message="tensor-product result exceeds the exact output envelope",
        )
    return order, classes, dimension


def character_tensor_product(
    request: CharacterTensorProductRequest,
) -> CharacterRingElement:
    """Return exact irreducible coordinates of a bounded virtual-character product.

    This currently authenticates complete canonical tables for the trivial,
    cyclic (order at most 60), and S3 groups. It makes no general-group claim.
    """
    if not isinstance(request, CharacterTensorProductRequest):
        raise _invalid(
            "groups.characters.tensor_product_request_type",
            "request must contain two table-bound virtual characters",
            ("request",),
        )
    left, right = request.left, request.right
    if not isinstance(left, CharacterRingElement) or not isinstance(
        right, CharacterRingElement
    ):
        raise _invalid(
            "groups.characters.tensor_product_input_type",
            "both inputs must be virtual-character ring elements",
            ("request",),
        )
    _admit_ring_element_shape(left, "left")
    _admit_ring_element_shape(right, "right")
    lt, rt = left.table, right.table
    if lt.partition.source != rt.partition.source:
        raise _invalid(
            "groups.characters.tensor_product_parent_mismatch",
            "both virtual characters must have the same concrete group parent",
            ("right", "table", "partition", "source"),
        )
    source = lt.partition.source
    actual_order, source_work = _admit_source_group_order(source)
    if actual_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("left", "table", "partition", "source"),
            code="groups.characters.tensor_product_group_order_exceeds_envelope",
            message="tensor products currently admit group order at most 60",
        )
    # Source order and its finite work are computed without trusting the
    # retained partition. Charge that work with the arithmetic below.
    _admit_tensor_arithmetic(left, right, source, source_work, actual_order)
    raw_classes = group_conjugacy_classes(
        source.degree, [list(g) for g in source.generators]
    )
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in raw_classes)
    )
    table = character_table(partition)
    if lt != table or rt != table:
        raise _invalid(
            "groups.characters.tensor_product_noncanonical_table",
            "both inputs must retain the exact canonical character table for their group",
            ("request",),
        )
    table_order = table.axis.cyclotomic_order
    classes = len(table.axis.class_sizes)
    dimension = euler_phi(table_order)

    def expand(element: CharacterRingElement) -> FiniteClassFunction:
        values = []
        for j in range(classes):
            coefficients = [Fraction(0) for _ in range(dimension)]
            for multiplicity, row in zip(
                element.irreducible_multiplicities, table.rows, strict=True
            ):
                for k, value in enumerate(row.values[j].coefficients):
                    coefficients[k] += multiplicity * value.as_fraction()
            values.append(_make_value(table_order, tuple(coefficients)))
        return FiniteClassFunction._from_kernel(axis=table.axis, values=tuple(values))

    product = class_function_pointwise_product(expand(left), expand(right))
    decomposed = _coordinates_on_authenticated_table(product, table)
    if decomposed.table != table:
        raise _invalid(
            "groups.characters.tensor_product_reconstruction",
            "product decomposition changed the canonical table",
            ("request",),
        )
    # Verify the returned coordinates reconstruct the exact valuewise product.
    if expand(decomposed) != product:
        raise _invalid(
            "groups.characters.tensor_product_reconstruction",
            "irreducible coordinates failed exact reconstruction",
            ("request",),
        )
    return decomposed


def _character_lambda_square(
    request: CharacterSymmetricSquareRequest | CharacterExteriorSquareRequest,
    *,
    adams_sign: int,
) -> CharacterRingElement:
    """Compute (x tensor x +/- psi^2(x))/2 in one admitted table basis."""
    if not isinstance(
        request, (CharacterSymmetricSquareRequest, CharacterExteriorSquareRequest)
    ):
        raise _invalid(
            "groups.characters.lambda_square_request_type",
            "request must contain one table-bound virtual character",
            ("request",),
        )
    element = request.character
    if not isinstance(element, CharacterRingElement):
        raise _invalid(
            "groups.characters.lambda_square_input_type",
            "input must be a table-bound virtual character",
            ("character",),
        )
    _admit_ring_element_shape(element, "character")
    source = element.table.partition.source
    actual_order, source_work = _admit_source_group_order(source)
    if actual_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("character", "table", "partition", "source"),
            code="groups.characters.lambda_square_group_order_exceeds_envelope",
            message="symmetric and exterior squares admit group order at most 60",
        )
    # Reuse the tensor admission's exact height, basis-pairing, reconstruction,
    # and serialized-result envelope, then charge for the class squaring map
    # and the addition/division used by the lambda identity.
    coordinate_digits = max(
        1, *(len(str(abs(value))) for value in element.irreducible_multiplicities)
    )
    table_digits = max(
        1,
        *(
            max(len(str(abs(value.num))), len(str(value.den)))
            for row in element.table.rows
            for class_value in row.values
            for value in class_value.coefficients
        ),
    )
    lambda_digits = coordinate_digits + table_digits + len(str(actual_order)) + 2
    lambda_work = (
        actual_order
        * euler_phi(actual_order)
        * euler_phi(actual_order)
        * lambda_digits**2
        + actual_order * source.degree * (max(1, actual_order.bit_length()) + 2)
        + actual_order * euler_phi(actual_order) * lambda_digits**2
    )
    order, classes, dimension = _admit_tensor_arithmetic(
        element,
        element,
        source,
        source_work,
        actual_order,
        additional_work=lambda_work,
    )

    raw_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in raw_classes)
    )
    table = character_table(partition)
    if element.table != table:
        raise _invalid(
            "groups.characters.lambda_square_noncanonical_table",
            "input must retain the exact canonical character table for its group",
            ("character", "table"),
        )
    order = table.axis.cyclotomic_order
    classes = len(table.axis.class_sizes)
    dimension = euler_phi(order)

    # Expand only after admission and table authentication. The class map is
    # derived from the complete canonical partition, so no caller-supplied
    # power-map claims enter the exact operation.
    def expand(value: CharacterRingElement) -> tuple[tuple[Fraction, ...], ...]:
        result: list[tuple[Fraction, ...]] = []
        for class_index in range(classes):
            coefficients = [Fraction(0) for _ in range(dimension)]
            for multiplicity, row in zip(
                value.irreducible_multiplicities, table.rows, strict=True
            ):
                for power, coefficient in enumerate(
                    row.values[class_index].coefficients
                ):
                    coefficients[power] += multiplicity * coefficient.as_fraction()
            result.append(tuple(coefficients))
        return tuple(result)

    values = expand(element)
    elements_by_class = {
        tuple(group_element): class_index
        for class_index, conjugacy_class in enumerate(table.partition.classes)
        for group_element in conjugacy_class
    }

    def square(permutation: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            permutation[permutation[index]] for index in range(len(permutation))
        )

    class_square_map = tuple(
        elements_by_class[square(tuple(conjugacy_class[0]))]
        for conjugacy_class in table.partition.classes
    )
    square_values = []
    for class_index, class_value in enumerate(values):
        tensor_square = multiply_values(order, class_value, class_value)
        adams_square = values[class_square_map[class_index]]
        signed_adams = scale_value(
            order,
            Fraction(adams_sign),
            adams_square,
        )
        numerator = add_values(order, tensor_square, signed_adams)
        square_values.append(scale_value(order, Fraction(1, 2), numerator))

    square_function = FiniteClassFunction._from_kernel(
        axis=table.axis,
        values=tuple(_make_value(order, value) for value in square_values),
    )
    result = _coordinates_on_authenticated_table(square_function, table)
    if expand(result) != tuple(square_values):
        raise _invalid(
            "groups.characters.lambda_square_reconstruction",
            "irreducible coordinates failed exact lambda-square reconstruction",
            ("request",),
        )
    return result


def character_symmetric_square(
    request: CharacterSymmetricSquareRequest,
) -> CharacterRingElement:
    """Return the exact second symmetric-power virtual character."""
    return _character_lambda_square(request, adams_sign=1)


def character_exterior_square(
    request: CharacterExteriorSquareRequest,
) -> CharacterRingElement:
    """Return the exact second exterior-power virtual character."""
    return _character_lambda_square(request, adams_sign=-1)


def _admit_character_kernel(
    element: CharacterRingElement,
    source: PermutationGroup,
    actual_order: int,
    source_work: int,
) -> None:
    """Admit character arithmetic, source enumeration, and returned groups."""
    coordinate_digits = max(
        1, *(len(str(abs(value))) for value in element.irreducible_multiplicities)
    )
    # A canonical irreducible value is a sum of at most sqrt(|G|) roots of
    # unity. This bound is independent of an untrusted serialized table.
    table_digits_bound = len(str(actual_order)) + 2
    kernel_work = (
        actual_order**2
        * euler_phi(actual_order)
        * coordinate_digits
        * table_digits_bound
        + actual_order**2 * source.degree * max(1, len(source.generators))
        + actual_order**3 * source.degree
        + MAX_CHARACTER_TABLE_CELLS
        + source_work
    )
    if kernel_work > MAX_CHARACTER_KERNEL_WORK:
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.kernel_work_exceeds_envelope",
            message="canonical-table validation and kernel generation exceed the work envelope",
        )
    generator_count_bound = max(1, actual_order.bit_length())
    output_bytes = (
        2_048 + (len(source.generators) + generator_count_bound) * source.degree * 24
    )
    if (
        output_bytes > MAX_CHARACTER_KERNEL_OUTPUT_BYTES
        or output_bytes > CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.kernel_output_exceeds_envelope",
            message="ambient group and kernel generators exceed the output envelope",
        )


def _character_values_on_classes(
    element: CharacterRingElement, table: CharacterTableResult
) -> tuple[tuple[Fraction, ...], ...]:
    """Expand an authenticated ordinary character on its exact class axis."""
    field_order = table.axis.cyclotomic_order
    class_count = len(table.partition.classes)
    phi_dimension = euler_phi(field_order)
    class_values: list[tuple[Fraction, ...]] = []
    for class_index in range(class_count):
        coefficients = [Fraction(0) for _ in range(phi_dimension)]
        for multiplicity, row in zip(
            element.irreducible_multiplicities, table.rows, strict=True
        ):
            for power, value in enumerate(row.values[class_index].coefficients):
                coefficients[power] += multiplicity * value.as_fraction()
        class_values.append(tuple(coefficients))

    identity = tuple(range(table.partition.source.degree))
    identity_class = next(
        index
        for index, conjugacy_class in enumerate(table.partition.classes)
        if identity in conjugacy_class
    )
    degree_value = class_values[identity_class]
    identity_degree = degree_value[0]
    if identity_degree.denominator != 1 or any(degree_value[1:]):
        raise _invalid(
            "groups.characters.kernel_invalid_degree",
            "canonical character has a nonintegral degree",
            ("character",),
        )
    return tuple(class_values)


def _permutation_compose(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    """Compose two permutations in image-list form."""
    return tuple(left[right[index]] for index in range(len(left)))


def _permutation_group_generated_by(
    source: PermutationGroup,
    subgroup_elements: set[tuple[int, ...]],
) -> PermutationGroup:
    """Return a deterministic generating set for an admitted subgroup."""
    identity = tuple(range(source.degree))

    def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        return _permutation_compose(left, right)

    def generated(generators: tuple[tuple[int, ...], ...]) -> set[tuple[int, ...]]:
        members = {identity}
        pending = [identity]
        while pending:
            current = pending.pop()
            for generator in generators:
                candidate = compose(current, generator)
                if candidate not in members:
                    members.add(candidate)
                    pending.append(candidate)
        return members

    if identity not in subgroup_elements:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    generators: list[tuple[int, ...]] = []
    members = {identity}
    for candidate in sorted(subgroup_elements):
        if candidate not in members:
            generators.append(candidate)
            members = generated(tuple(generators))
    if members != subgroup_elements:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    if not generators:
        generators = [identity]
    return PermutationGroup(degree=source.degree, generators=tuple(generators))


def character_kernel(request: CharacterKernelRequest) -> CharacterKernel:
    """Return the exact kernel of a supported ordinary virtual-table character.

    The accepted coordinates must be nonnegative, so they describe an actual
    finite-dimensional representation. For a finite-dimensional unitary
    representation, ``chi(g) == chi(1)`` exactly iff every eigenvalue of
    ``rho(g)`` is 1, hence iff ``g`` is in the kernel.
    """
    if not isinstance(request, CharacterKernelRequest):
        raise _invalid(
            "groups.characters.kernel_request_type",
            "request must contain one table-bound ordinary character",
            ("request",),
        )
    element = request.character
    if not isinstance(element, CharacterRingElement):
        raise _invalid(
            "groups.characters.kernel_input_type",
            "input must be a table-bound ordinary character",
            ("character",),
        )
    _admit_ring_element_shape(element, "character")
    if any(multiplicity < 0 for multiplicity in element.irreducible_multiplicities):
        raise _invalid(
            "groups.characters.kernel_requires_ordinary_character",
            "kernel is defined here only for nonnegative irreducible multiplicities",
            ("character", "irreducible_multiplicities"),
        )
    source = element.table.partition.source
    actual_order, source_work = _admit_source_group_order(source)
    if actual_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("character", "table", "partition", "source"),
            code="groups.characters.kernel_group_order_exceeds_envelope",
            message="character kernels admit groups of order at most 60",
        )
    _admit_character_kernel(element, source, actual_order, source_work)
    raw_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in raw_classes)
    )
    table = character_table(partition)
    if element.table != table:
        raise _invalid(
            "groups.characters.kernel_noncanonical_table",
            "input must retain the exact canonical character table for its group",
            ("character", "table"),
        )
    class_values = _character_values_on_classes(element, table)
    identity_class = next(
        index
        for index, conjugacy_class in enumerate(table.partition.classes)
        if tuple(range(source.degree)) in conjugacy_class
    )
    degree_value = class_values[identity_class]
    kernel_elements = {
        tuple(group_element)
        for class_index, value in enumerate(class_values)
        if value == degree_value
        for group_element in table.partition.classes[class_index]
    }
    subgroup = _permutation_group_generated_by(source, kernel_elements)
    return CharacterKernel._from_kernel(ambient_group=source, subgroup=subgroup)


def _admit_character_center(
    element: CharacterRingElement,
    source: PermutationGroup,
    actual_order: int,
    source_work: int,
) -> None:
    """Admit table expansion, exact character norms, and subgroup output."""
    coordinate_digits = max(
        1, *(len(str(abs(value))) for value in element.irreducible_multiplicities)
    )
    field_degree = euler_phi(actual_order)
    table_digits_bound = len(str(actual_order)) + 2
    class_value_digits = coordinate_digits + table_digits_bound
    work = (
        actual_order**2 * field_degree * coordinate_digits * table_digits_bound
        + actual_order * field_degree**2 * class_value_digits**2
        + actual_order**2 * source.degree * max(1, len(source.generators))
        + actual_order**3 * source.degree
        + MAX_CHARACTER_TABLE_CELLS
        + source_work
    )
    if work > MAX_CHARACTER_CENTER_WORK:
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.center_work_exceeds_envelope",
            message="canonical-table validation and exact center computation exceed the work envelope",
        )
    # The result retains the full canonical character table. Bound its
    # coefficient payload, not only the selected scalar values: each class
    # value has field_degree rational coefficients and there are at most
    # actual_order classes. Rational JSON encoding needs numerator and
    # denominator digits plus separators; use a conservative per-digit cost.
    table_bytes = (
        actual_order**2
        * field_degree
        * (coordinate_digits + table_digits_bound + 8)
    )
    output_bytes = (
        2_048
        + table_bytes
        + (len(source.generators) + max(1, actual_order.bit_length()))
        * source.degree
        * 24
        + actual_order * field_degree * (coordinate_digits + table_digits_bound) * 8
    )
    if (
        output_bytes > MAX_CHARACTER_CENTER_OUTPUT_BYTES
        or output_bytes > CanonicalLimits().max_output_bytes
    ):
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.center_output_exceeds_envelope",
            message="character center values and subgroup exceed the output envelope",
        )


def character_center(request: CharacterCenterRequest) -> CharacterCenter:
    r"""Return the subgroup on which an ordinary representation is scalar.

    A finite-dimensional complex representation can be made unitary. For
    each group element, ``chi(g) * conjugate(chi(g)) = chi(1)^2`` exactly
    iff all eigenvalues of its representing matrix agree, so its normalized
    trace is a root of unity and the element acts as a scalar.
    """
    if not isinstance(request, CharacterCenterRequest):
        raise _invalid(
            "groups.characters.center_request_type",
            "request must contain one table-bound ordinary character",
            ("request",),
        )
    element = request.character
    if not isinstance(element, CharacterRingElement):
        raise _invalid(
            "groups.characters.center_input_type",
            "input must be a table-bound ordinary character",
            ("character",),
        )
    _admit_ring_element_shape(element, "character")
    if any(multiplicity < 0 for multiplicity in element.irreducible_multiplicities):
        raise _invalid(
            "groups.characters.center_requires_ordinary_character",
            "character center is defined here only for nonnegative irreducible multiplicities",
            ("character", "irreducible_multiplicities"),
        )
    source = element.table.partition.source
    actual_order, source_work = _admit_source_group_order(source)
    if actual_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("character", "table", "partition", "source"),
            code="groups.characters.center_group_order_exceeds_envelope",
            message="character centers admit groups of order at most 60",
        )
    _admit_character_center(element, source, actual_order, source_work)
    raw_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    partition = GroupConjugacyClassesResult._from_kernel(
        source, tuple(tuple(tuple(g) for g in cls) for cls in raw_classes)
    )
    table = character_table(partition)
    if element.table != table:
        raise _invalid(
            "groups.characters.center_noncanonical_table",
            "input must retain the exact canonical character table for its group",
            ("character", "table"),
        )
    class_values = _character_values_on_classes(element, table)
    identity = tuple(range(source.degree))
    identity_class = next(
        index
        for index, conjugacy_class in enumerate(table.partition.classes)
        if identity in conjugacy_class
    )
    degree_value = class_values[identity_class]
    degree = degree_value[0]
    if degree.denominator != 1 or any(degree_value[1:]) or degree <= 0:
        raise _invalid(
            "groups.characters.center_invalid_degree",
            "canonical ordinary character must have a positive integral degree",
            ("character",),
        )

    field_order = table.axis.cyclotomic_order
    target_norm = (degree * degree,) + (Fraction(0),) * (euler_phi(field_order) - 1)
    selected_indices: list[int] = []
    scalar_values: list[CyclotomicValue] = []
    scalar_classes: set[tuple[int, ...]] = set()
    for class_index, class_value in enumerate(class_values):
        norm = multiply_values(
            field_order,
            class_value,
            conjugate_value(field_order, class_value),
        )
        if norm != target_norm:
            continue
        selected_indices.append(class_index)
        scalar_values.append(
            _make_value(
                field_order,
                scale_value(field_order, Fraction(1, degree), class_value),
            )
        )
        scalar_classes.update(
            tuple(group_element)
            for group_element in table.partition.classes[class_index]
        )
    subgroup = _permutation_group_generated_by(source, scalar_classes)
    return CharacterCenter._from_kernel(
        character=element,
        subgroup=subgroup,
        scalar_class_indices=tuple(selected_indices),
        scalar_values=tuple(scalar_values),
    )


__all__ = [
    "MAX_CHARACTER_CENTER_OUTPUT_BYTES",
    "MAX_CHARACTER_CENTER_WORK",
    "MAX_CHARACTER_KERNEL_OUTPUT_BYTES",
    "MAX_CHARACTER_KERNEL_WORK",
    "MAX_CHARACTER_RING_DECOMPOSITION_OUTPUT_BYTES",
    "MAX_CHARACTER_RING_DECOMPOSITION_WORK",
    "MAX_CHARACTER_TENSOR_PRODUCT_WORK",
    "character_center",
    "character_exterior_square",
    "character_kernel",
    "character_symmetric_square",
    "character_tensor_product",
    "class_function_character_decomposition",
]
