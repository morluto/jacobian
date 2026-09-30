"""Exact degree evaluation for bounded finite-group characters."""

from __future__ import annotations

from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups._models import (
    MAX_GROUP_DEGREE,
    PermutationGroup,
)
from jacobian.math.groups.characters._models import (
    MAX_CLASS_COUNT,
    CharacterRingElement,
    ClassAxis,
)
from jacobian.math.groups.characters.degree._models import (
    CharacterDegree,
)
from jacobian.math.groups.characters.representation_ring_operations import (
    _admit_ring_element_shape,
)

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


def character_degree(character: CharacterRingElement) -> CharacterDegree:
    """Return chi(1) for a table-bound ordinary finite-group character.

    For an ordinary character, chi(1) = sum_i m_i * d_i, where the m_i are the
    irreducible multiplicities and the d_i are the irreducible degrees carried
    by the retained canonical table's rows. Both are fields of values other
    operations already publish, so this is the native projection of that dot
    product and it stays out of the catalog: it adds no postcondition that
    `finite_group.character_table.compute` and the character-ring decomposition
    do not already establish.

    The earlier implementation also re-derived the source group, its conjugacy
    classes, and the whole character table purely to confirm that the caller's
    retained copy was canonical. That made the operation strictly more expensive
    than the projection it replaces, so the re-derivation is gone; the shape
    checks below are what this projection actually relies on.
    """
    if not isinstance(character, CharacterRingElement):
        _invalid(
            "groups.characters.degree_input_type",
            "input must be a table-bound finite-group character",
            ("character",),
        )
    # `model_construct` populates a carrier without running the table's own
    # validators, so a caller could otherwise present a table whose rows do not
    # describe a group. Re-validating is linear in the retained table and
    # replaces the group, conjugacy-class, and table re-derivation this used to
    # perform for the same purpose.
    try:
        character = CharacterRingElement.model_validate(
            character.model_dump(mode="python", warnings=False)
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("character",),
            code="groups.characters.degree_noncanonical_table",
            message="input must retain the exact canonical character table for its group",
        ) from exc
    _admit_ring_element_shape(character, "character")
    _admit_retained_table_shape(character)
    multiplicities = character.irreducible_multiplicities
    if len(multiplicities) != len(character.table.rows):
        _invalid(
            "groups.characters.degree_row_alignment",
            "one multiplicity per irreducible character row is required",
            ("character", "irreducible_multiplicities"),
        )
    if any(value < 0 for value in multiplicities):
        _invalid(
            "groups.characters.degree_requires_ordinary_character",
            "character degree is defined for nonnegative irreducible multiplicities",
            ("character", "irreducible_multiplicities"),
        )
    value = sum(
        multiplicity * row.degree
        for multiplicity, row in zip(multiplicities, character.table.rows, strict=True)
    )
    if abs(value) > MAX_GROUP_DEGREE:
        raise OperationResourceAdmissionError(
            location=("character",),
            code="groups.characters.degree_output_exceeds_envelope",
            message="source-bound character degree exceeds the exact output envelope",
        )
    return CharacterDegree(character=character, degree=value)


__all__ = ["character_degree"]
