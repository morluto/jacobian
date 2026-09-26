"""Exact transport of Dirichlet characters between supplied unit bases."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.coordinate_basis._models import (
    DirichletCharacterBasisChangeResult,
    DirichletCharacterCoordinateIsomorphism,
)
from jacobian.math.number_theory.characters.operations import (
    _require_character,
    require_complete_character_group,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
)

_MAX_WORK = 131_072


def _domain_error(code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=("coordinate_isomorphism",),
        code=f"dirichlet_character.coordinate_basis.{code}",
        message=message,
    )


def _resource_error(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("coordinate_isomorphism",),
        code=f"dirichlet_character.coordinate_basis.{code}",
        message=message,
    )


def _require_coordinate_isomorphism(
    coordinate_isomorphism: DirichletCharacterCoordinateIsomorphism,
) -> DirichletCharacterCoordinateIsomorphism:
    """Admit a native isomorphism argument before any field is dereferenced."""

    if not isinstance(coordinate_isomorphism, DirichletCharacterCoordinateIsomorphism):
        _domain_error(
            "isomorphism_type",
            "coordinate_isomorphism must be a coordinate-basis isomorphism value",
        )
    try:
        return DirichletCharacterCoordinateIsomorphism.model_validate(
            coordinate_isomorphism.model_dump()
        )
    except (ValidationError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("coordinate_isomorphism",),
            code="dirichlet_character.coordinate_basis.isomorphism_invalid",
            message="coordinate isomorphism has malformed authored fields",
        ) from exc


def _character_exponent(character: DirichletCharacter, row: tuple[int, ...]) -> int:
    group = character.group
    return (
        sum(
            coordinate * (group.exponent // order) * unit_coordinate
            for coordinate, order, unit_coordinate in zip(
                character.coordinates, group.generator_orders, row, strict=True
            )
        )
        % group.exponent
    )


def change_dirichlet_character_coordinate_basis(
    character: DirichletCharacter,
    coordinate_isomorphism: DirichletCharacterCoordinateIsomorphism,
) -> DirichletCharacterBasisChangeResult:
    """Express the same residue character in a supplied target unit basis."""

    coordinate_isomorphism = _require_coordinate_isomorphism(coordinate_isomorphism)
    character = _require_character(character)
    source = coordinate_isomorphism.source_group
    target = coordinate_isomorphism.target_group
    if character.group != source:
        _domain_error(
            "source_mismatch", "isomorphism source group must equal character group"
        )
    if source.modulus != target.modulus:
        _domain_error("modulus_mismatch", "coordinate bases must use one modulus")
    if source.exponent != target.exponent:
        _domain_error(
            "cyclotomic_parent_mismatch",
            "source and target bases must use the same exact root-of-unity parent",
        )

    work = (
        len(source.unit_residues)
        * (len(source.generator_orders) + len(target.generator_orders))
        + source.modulus
    )
    if work > _MAX_WORK:
        _resource_error("work_bound", "coordinate transport exceeds its admitted work")
    if source.modulus > MAX_CHARACTER_GROUP_MODULUS:
        _resource_error("modulus_bound", "modulus exceeds the character table bound")

    require_complete_character_group(source)
    require_complete_character_group(target)

    source_rows = dict(zip(source.unit_residues, source.unit_coordinates, strict=True))
    for index, generator in enumerate(target.generators):
        supplied_image = (
            coordinate_isomorphism.target_generator_images_in_source_coordinates[index]
        )
        if supplied_image != source_rows[generator]:
            _domain_error(
                "isomorphism_image_mismatch",
                "each target generator image must name that same unit in source coordinates",
            )

    target_coordinates: list[int] = []
    for row in coordinate_isomorphism.target_generator_images_in_source_coordinates:
        exponent = _character_exponent(character, row)
        order_factor = (
            source.exponent // target.generator_orders[len(target_coordinates)]
        )
        if exponent % order_factor:
            _domain_error(
                "character_image_order",
                "character value on a target generator does not have its declared order",
            )
        target_coordinates.append(exponent // order_factor)

    transported = DirichletCharacter(
        group=target, coordinates=tuple(target_coordinates)
    )
    residues = tuple(range(source.modulus))
    target_rows = dict(zip(target.unit_residues, target.unit_coordinates, strict=True))
    values = tuple(
        CyclotomicValue(
            order=target.exponent,
            exponent=_character_exponent(transported, target_rows[residue]),
        )
        if residue in target_rows
        else None
        for residue in residues
    )
    return DirichletCharacterBasisChangeResult(
        source_character=character,
        transported_character=transported,
        residues=residues,
        values=values,
    )


__all__ = ["change_dirichlet_character_coordinate_basis"]
