"""Exact degree evaluation for bounded finite-group characters."""

from __future__ import annotations

from typing import NoReturn

from jacobian._exact import CanonicalRational
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
from jacobian.math.groups.characters._cyclotomic import euler_phi
from jacobian.math.groups.characters._models import (
    MAX_CHARACTER_TABLE_CELLS,
    MAX_CLASS_COUNT,
    MAX_CYCLOTOMIC_ORDER,
    MAX_VALUE_COEFFICIENT_DIGITS,
    CharacterRingElement,
    ClassAxis,
    ConjugacyClassPartition,
)
from jacobian.math.groups.characters.degree._models import (
    CharacterDegree,
    CharacterDegreeRequest,
)
from jacobian.math.groups.characters.operations import character_table
from jacobian.math.groups.characters.representation_ring_operations import (
    _admit_ring_element_shape,
    _admit_source_group_order,
    _admit_tensor_partition_shape,
)
from jacobian.math.groups.operations import group_conjugacy_classes

MAX_CHARACTER_DEGREE_WORK = 50_000_000


def _invalid(code: str, message: str, location: tuple[str, ...]) -> NoReturn:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _decimal_digit_upper_bound(value: int) -> int:
    bits = abs(value).bit_length()
    return 1 if bits == 0 else (bits * 30_103) // 100_000 + 2


def _admit_retained_table_shape(character: CharacterRingElement) -> None:
    """Bound table metadata that the shared operand check does not inspect."""
    table = character.table
    for index, row in enumerate(table.rows):
        if type(row.label) is not str or not 1 <= len(row.label) <= 64:
            _invalid(
                "groups.characters.degree_row_label",
                "character row labels must contain 1 to 64 characters",
                ("character", "table", "rows", str(index), "label"),
            )
        if type(row.degree) is not int or row.degree.bit_length() > 20:
            _invalid(
                "groups.characters.degree_row_shape",
                "irreducible row degrees must be positive admitted integers",
                ("character", "table", "rows", str(index), "degree"),
            )
    if (
        type(table.degree_square_sum) is not int
        or table.degree_square_sum.bit_length() > 20
    ):
        _invalid(
            "groups.characters.degree_table_shape",
            "character-table degree square sum must be a positive bounded integer",
            ("character", "table", "degree_square_sum"),
        )

    axis: object = getattr(table, "axis", None)
    if not isinstance(axis, ClassAxis):
        _invalid(
            "groups.characters.degree_axis",
            "character table must retain a bounded class axis",
            ("character", "table", "axis"),
        )
    class_sizes = getattr(axis, "class_sizes", None)
    if not isinstance(class_sizes, tuple) or len(class_sizes) > MAX_CLASS_COUNT:
        _invalid(
            "groups.characters.degree_axis",
            "class-axis sizes must use a bounded tuple",
            ("character", "table", "axis", "class_sizes"),
        )
    if any(
        type(size) is not int or abs(size).bit_length() > 20 for size in class_sizes
    ):
        _invalid(
            "groups.characters.degree_axis",
            "class-axis sizes must be bounded integers",
            ("character", "table", "axis", "class_sizes"),
        )
    if (
        type(axis.group_order) is not int
        or abs(axis.group_order).bit_length() > 20
        or type(axis.cyclotomic_order) is not int
        or abs(axis.cyclotomic_order).bit_length() > 20
    ):
        _invalid(
            "groups.characters.degree_axis",
            "class-axis group and cyclotomic orders must be admitted integers",
            ("character", "table", "axis"),
        )
    source = table.partition.source
    axis_group = axis.group
    if not isinstance(axis_group, PermutationGroup):
        _invalid(
            "groups.characters.degree_axis",
            "class axis must retain its concrete source group",
            ("character", "table", "axis", "group"),
        )
    if (
        type(axis_group.degree) is not int
        or not 1 <= axis_group.degree <= MAX_GROUP_DEGREE
        or not isinstance(axis_group.generators, tuple)
        or len(axis_group.generators) > MAX_GROUP_DEGREE
        or any(
            not isinstance(generator, tuple)
            or len(generator) != axis_group.degree
            or any(
                type(point) is not int or not 0 <= point < axis_group.degree
                for point in generator
            )
            for generator in axis_group.generators
        )
    ):
        _invalid(
            "groups.characters.degree_axis",
            "class-axis source group exceeds its permutation envelope",
            ("character", "table", "axis", "group"),
        )
    if axis_group != source:
        _invalid(
            "groups.characters.degree_axis",
            "class axis must retain the character table source group",
            ("character", "table", "axis", "group"),
        )
    representatives = axis.class_representatives
    if (
        not isinstance(representatives, tuple)
        or len(representatives) > MAX_CLASS_COUNT
        or any(
            not isinstance(representative, tuple)
            or len(representative) > MAX_GROUP_DEGREE
            or any(
                type(point) is not int or abs(point).bit_length() > 20
                for point in representative
            )
            for representative in representatives
        )
    ):
        _invalid(
            "groups.characters.degree_axis",
            "class-axis representatives must match the bounded class partition",
            ("character", "table", "axis", "class_representatives"),
        )


def _preflight_character_scan(
    character: CharacterRingElement, *, group_order: int
) -> int:
    """Bound the table scan before exact coefficient fields are inspected."""
    coordinates = getattr(character, "irreducible_multiplicities", None)
    if not isinstance(coordinates, tuple) or len(coordinates) > MAX_CLASS_COUNT:
        _invalid(
            "groups.characters.degree_coordinate_shape",
            "ordinary character coordinates must be a bounded tuple",
            ("character", "irreducible_multiplicities"),
        )
    coordinate_digit_work = 0
    for index, coordinate in enumerate(coordinates):
        if type(coordinate) is not int or coordinate.bit_length() > 1702:
            raise OperationResourceAdmissionError(
                location=("character", "irreducible_multiplicities", str(index)),
                code="groups.characters.degree_coordinate_height",
                message="character coordinates exceed the exact coefficient envelope",
            )
        coordinate_digit_work += _decimal_digit_upper_bound(coordinate)
    table = getattr(character, "table", None)
    rows = getattr(table, "rows", None)
    if not isinstance(rows, tuple) or not 1 <= len(rows) <= min(
        MAX_CLASS_COUNT, group_order
    ):
        _invalid(
            "groups.characters.degree_table_shape",
            "character table must have a bounded nonempty row tuple",
            ("character", "table", "rows"),
        )
    table_cells = 0
    coefficient_count = 0
    coefficient_digit_work = 0
    for row_index, row in enumerate(rows):
        values = getattr(row, "values", None)
        if not isinstance(values, tuple) or not 1 <= len(values) <= min(
            MAX_CLASS_COUNT, group_order
        ):
            _invalid(
                "groups.characters.degree_table_shape",
                "each character row must have a bounded nonempty value tuple",
                ("character", "table", "rows", str(row_index), "values"),
            )
        table_cells += len(values)
        if table_cells > MAX_CHARACTER_TABLE_CELLS:
            raise OperationResourceAdmissionError(
                location=("character", "table", "rows"),
                code="groups.characters.degree_table_exceeds_envelope",
                message="character table exceeds the exact cell envelope",
            )
        for value_index, value in enumerate(values):
            coefficients = getattr(value, "coefficients", None)
            if not isinstance(coefficients, tuple) or len(coefficients) != euler_phi(
                group_order
            ):
                _invalid(
                    "groups.characters.degree_table_shape",
                    "cyclotomic coefficient vectors must have bounded length",
                    (
                        "character",
                        "table",
                        "rows",
                        str(row_index),
                        "values",
                        str(value_index),
                    ),
                )
            coefficient_count += len(coefficients)
            for coefficient in coefficients:
                if (
                    not isinstance(coefficient, CanonicalRational)
                    or type(coefficient.num) is not int
                    or type(coefficient.den) is not int
                    or coefficient.den <= 0
                    or coefficient.num.bit_length() > 1702
                    or coefficient.den.bit_length() > 1702
                ):
                    _invalid(
                        "groups.characters.degree_coefficient_shape",
                        "character table coefficients must fit the exact rational envelope",
                        (
                            "character",
                            "table",
                            "rows",
                            str(row_index),
                            "values",
                            str(value_index),
                        ),
                    )
                coefficient_digit_work += _decimal_digit_upper_bound(coefficient.num)
                coefficient_digit_work += _decimal_digit_upper_bound(coefficient.den)
    scan_work = 4 * table_cells + 8 * coefficient_count + coefficient_digit_work
    # Include the bounded permutation/class shape pass performed by the shared
    # ring-element admission helper that follows this raw size probe.
    scan_work += MAX_GROUP_DEGREE**3
    if scan_work > MAX_CHARACTER_DEGREE_WORK:
        raise OperationResourceAdmissionError(
            location=("character", "table"),
            code="groups.characters.degree_work_exceeds_envelope",
            message="character table validation exceeds the admitted scan work",
        )
    return scan_work


def _admit_degree_work_and_output(
    character: CharacterRingElement,
    *,
    group_order: int,
    source_work: int,
    scan_work: int,
) -> None:
    source = character.table.partition.source
    degree = source.degree
    generator_count = len(source.generators)
    cyclotomic_dimension = euler_phi(group_order)
    table_cells = group_order * group_order * cyclotomic_dimension
    partition_cells = group_order * degree
    # Character-table reconstruction uses a bounded group conjugacy partition,
    # followed by at most order^2 * phi(order) exact cells.
    work = (
        scan_work
        + source_work
        + 2 * group_order**2 * degree * max(1, generator_count)
        + table_cells * 4
        + group_order
    )
    if table_cells > MAX_CHARACTER_TABLE_CELLS:
        raise OperationResourceAdmissionError(
            location=("character", "table"),
            code="groups.characters.degree_table_exceeds_envelope",
            message="canonical character table exceeds the exact cell envelope",
        )
    if work > MAX_CHARACTER_DEGREE_WORK:
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.degree_work_exceeds_envelope",
            message="character table authentication and degree evaluation exceed the work bound",
        )

    # The result retains the canonical table, not the caller's table claim.
    # Each exact rational coefficient has bounded fields and separators; the
    # complete table cell cap therefore bounds its serialized contribution.
    output_bytes = (
        1024
        + 2 * generator_count * (degree * 4 + 32)
        + partition_cells * 8
        + group_order * 128
        + table_cells * 64
        + group_order * (MAX_VALUE_COEFFICIENT_DIGITS + 64)
    )
    if output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.degree_output_exceeds_envelope",
            message="source-bound character degree exceeds the exact output envelope",
        )


def character_degree(request: CharacterDegreeRequest) -> CharacterDegree:
    """Return chi(1) for a table-bound ordinary finite-group character."""
    if not isinstance(request, CharacterDegreeRequest):
        _invalid(
            "groups.characters.degree_request_type",
            "request must contain one table-bound ordinary character",
            ("request",),
        )
    character = getattr(request, "character", None)
    if not isinstance(character, CharacterRingElement):
        _invalid(
            "groups.characters.degree_input_type",
            "input must be a table-bound finite-group character",
            ("character",),
        )
    partition = getattr(getattr(character, "table", None), "partition", None)
    if not isinstance(partition, ConjugacyClassPartition):
        _invalid(
            "groups.characters.degree_partition_type",
            "character table must retain a concrete conjugacy partition",
            ("character", "table", "partition"),
        )
    source = _admit_tensor_partition_shape(partition, "character")
    group_order, source_work = _admit_source_group_order(source)
    if group_order > MAX_CYCLOTOMIC_ORDER:
        raise OperationResourceAdmissionError(
            location=("character", "table", "partition", "source"),
            code="groups.characters.degree_group_order_exceeds_envelope",
            message="character degree currently admits groups of order at most 60",
        )
    scan_work = _preflight_character_scan(character, group_order=group_order)
    _admit_ring_element_shape(character, "character")
    _admit_retained_table_shape(character)
    if any(value < 0 for value in character.irreducible_multiplicities):
        _invalid(
            "groups.characters.degree_requires_ordinary_character",
            "character degree is defined for nonnegative irreducible multiplicities",
            ("character", "irreducible_multiplicities"),
        )
    _admit_degree_work_and_output(
        character,
        group_order=group_order,
        source_work=source_work,
        scan_work=scan_work,
    )

    raw_classes = group_conjugacy_classes(
        source.degree, [list(generator) for generator in source.generators]
    )
    canonical_partition = GroupConjugacyClassesResult._from_kernel(
        source,
        tuple(
            tuple(tuple(group_element) for group_element in cls) for cls in raw_classes
        ),
    )
    table = character_table(canonical_partition)
    if character.table != table:
        _invalid(
            "groups.characters.degree_noncanonical_table",
            "input must retain the exact canonical character table for its group",
            ("character", "table"),
        )

    canonical_character = CharacterRingElement._from_kernel(
        table=table,
        irreducible_multiplicities=character.irreducible_multiplicities,
    )
    value = sum(
        multiplicity * row.degree
        for multiplicity, row in zip(
            canonical_character.irreducible_multiplicities, table.rows, strict=True
        )
    )
    return CharacterDegree(character=canonical_character, degree=value)


__all__ = ["MAX_CHARACTER_DEGREE_WORK", "character_degree"]
