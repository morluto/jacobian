"""Exact structural maps between supported modular-form spaces."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms.values import (
    MAX_MODULAR_CHARACTER_INCLUSION_LEVEL,
    ModularCharacterSpaceInclusion,
    ModularFormSpace,
)

MAX_CHARACTER_INCLUSION_WORK = 100_000


def _domain(
    message: str, *, code: str = "modular_form.character_inclusion_invalid"
) -> None:
    raise OperationDomainValidationError(
        location=("inclusion",), code=code, message=message
    )


def _character_angle(character: DirichletCharacter, residue: int) -> Fraction | None:
    """Return the exact rational angle of a character value on a unit."""

    group = character.group
    try:
        row_index = group.unit_residues.index(residue % group.modulus)
    except ValueError:
        return None
    row = group.unit_coordinates[row_index]
    exponent = (
        sum(
            coordinate * (group.exponent // axis_order) * unit_coordinate
            for coordinate, axis_order, unit_coordinate in zip(
                character.coordinates, group.generator_orders, row, strict=True
            )
        )
        % group.exponent
    )
    return Fraction(exponent, group.exponent)


def require_modular_character_space_inclusion(
    inclusion: ModularCharacterSpaceInclusion,
) -> ModularCharacterSpaceInclusion:
    """Re-establish the exact character-inflation relation of a supplied map."""

    if type(inclusion) is not ModularCharacterSpaceInclusion:
        _domain("a canonical modular-character space inclusion is required")
    try:
        canonical = ModularCharacterSpaceInclusion.model_validate(
            inclusion.model_dump()
        )
    except (ValidationError, AttributeError, TypeError, ValueError) as error:
        raise OperationDomainValidationError(
            location=("inclusion",),
            code="modular_form.character_inclusion_invalid",
            message="the modular-character space inclusion is malformed",
        ) from error

    source = canonical.source_space
    target = canonical.target_space
    if (
        type(source) is not ModularFormSpace
        or type(target) is not ModularFormSpace
        or type(source.character) is not DirichletCharacter
        or type(target.character) is not DirichletCharacter
    ):
        _domain("both inclusion parents must retain exact Dirichlet characters")
    source_level = source.level
    target_level = target.level
    if (
        type(source_level) is not int
        or type(target_level) is not int
        or source_level > MAX_MODULAR_CHARACTER_INCLUSION_LEVEL
        or target_level > MAX_MODULAR_CHARACTER_INCLUSION_LEVEL
    ):
        raise OperationResourceAdmissionError(
            location=("target_space", "level"),
            code="modular_form.character_inclusion_level_bound",
            message=(
                "character-space inclusion levels exceed the exact admitted bound "
                f"{MAX_MODULAR_CHARACTER_INCLUSION_LEVEL}"
            ),
        )

    source_character = cast(DirichletCharacter, source.character)
    target_character = cast(DirichletCharacter, target.character)
    # ModularFormSpace canonicalization validates each supplied finite-unit
    # presentation. This check is the sole additional character-map pass.
    target_units = target_character.group.unit_residues
    source_rank = len(source_character.group.generator_orders)
    target_rank = len(target_character.group.generator_orders)
    work = (
        source_level
        + target_level
        + len(target_units) * (source_rank + target_rank + 2)
    )
    if work > MAX_CHARACTER_INCLUSION_WORK:
        raise OperationResourceAdmissionError(
            location=("target_space",),
            code="modular_form.character_inclusion_work_bound",
            message="exact character-inclusion comparison exceeds its admitted work bound",
        )

    for residue in target_units:
        source_angle = _character_angle(source_character, residue)
        target_angle = _character_angle(target_character, residue)
        if source_angle is None or target_angle is None or source_angle != target_angle:
            _domain(
                "target character must equal the source character pulled back along reduction of units",
                code="modular_form.character_inflation_mismatch",
            )
    return canonical


def modular_form_character_space_inclusion(
    source_space: ModularFormSpace,
    target_space: ModularFormSpace,
) -> ModularCharacterSpaceInclusion:
    """Construct a nested-level inclusion with exact character inflation."""

    if (
        type(source_space) is not ModularFormSpace
        or type(target_space) is not ModularFormSpace
    ):
        _domain(
            "source and target must be exact modular-form space values",
            code="modular_form.character_inclusion_space_type",
        )
    inclusion = ModularCharacterSpaceInclusion.model_construct(
        map_kind="gamma0_character_inflation",
        source_space=source_space,
        target_space=target_space,
    )
    return require_modular_character_space_inclusion(inclusion)


__all__ = [
    "modular_form_character_space_inclusion",
    "require_modular_character_space_inclusion",
]
