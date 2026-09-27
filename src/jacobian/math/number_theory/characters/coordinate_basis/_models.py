"""Typed contracts for exact changes of Dirichlet-character coordinates."""

from __future__ import annotations

from math import gcd
from typing import Annotated

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterGroup,
)

CoordinateRow = Annotated[tuple[StrictInt, ...], Field(max_length=32)]
CoordinateImageRows = tuple[CoordinateRow, ...]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(
        f"dirichlet_character.coordinate_basis.{reason}", message
    )


class DirichletCharacterCoordinateIsomorphism(StrictModel):
    """A supplied unit-coordinate isomorphism from target basis to source basis.

    Each row gives the source unit coordinates of the corresponding target
    generator. Generator ordering is the one declared by each group value.
    """

    source_group: DirichletCharacterGroup
    target_group: DirichletCharacterGroup
    target_generator_images_in_source_coordinates: CoordinateImageRows = Field(
        max_length=32
    )

    @model_validator(mode="after")
    def require_bounded_matrix_shape(self) -> DirichletCharacterCoordinateIsomorphism:
        if len(self.target_generator_images_in_source_coordinates) != len(
            self.target_group.generators
        ):
            raise _validation_error(
                "generator_image_count",
                "one source-coordinate image is required for each target generator",
            )
        source_orders = self.source_group.generator_orders
        for image in self.target_generator_images_in_source_coordinates:
            if len(image) != len(source_orders):
                raise _validation_error(
                    "image_rank",
                    "each target-generator image must match the source coordinate rank",
                )
            if any(
                coordinate < 0 or coordinate >= order
                for coordinate, order in zip(image, source_orders, strict=True)
            ):
                raise _validation_error(
                    "image_range",
                    "source-coordinate images must lie in their cyclic factors",
                )
        return self


class DirichletCharacterBasisChangeRequest(StrictModel):
    character: DirichletCharacter
    coordinate_isomorphism: DirichletCharacterCoordinateIsomorphism


class DirichletCharacterBasisChangeResult(StrictModel):
    source_character: DirichletCharacter
    transported_character: DirichletCharacter
    residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    values: tuple[CyclotomicValue | None, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )

    @model_validator(mode="after")
    def require_complete_canonical_table(self) -> DirichletCharacterBasisChangeResult:
        modulus = self.source_character.group.modulus
        if self.residues != tuple(range(modulus)) or len(self.values) != modulus:
            raise _validation_error(
                "result_table_shape",
                "the result table must cover residues 0 through modulus minus one",
            )
        source = self.source_character
        transported = self.transported_character
        if (
            transported.group.modulus != modulus
            or transported.group.exponent != source.group.exponent
        ):
            raise _validation_error(
                "result_character_parent",
                "source and transported characters must share the modulus and cyclotomic parent",
            )

        def character_table(character: DirichletCharacter) -> tuple[CyclotomicValue | None, ...]:
            group = character.group
            unit_rows = dict(zip(group.unit_residues, group.unit_coordinates, strict=True))
            if tuple(group.unit_residues) != tuple(
                residue for residue in range(modulus) if gcd(residue, modulus) == 1
            ):
                raise _validation_error(
                    "result_character_units",
                    "character groups must retain the complete canonical unit residues",
                )
            table: list[CyclotomicValue | None] = []
            for residue in range(modulus):
                row = unit_rows.get(residue)
                if row is None:
                    table.append(None)
                    continue
                exponent = sum(
                    coordinate * (group.exponent // order) * unit_coordinate
                    for coordinate, order, unit_coordinate in zip(
                        character.coordinates, group.generator_orders, row, strict=True
                    )
                ) % group.exponent
                table.append(CyclotomicValue(order=group.exponent, exponent=exponent))
            return tuple(table)

        source_values = character_table(source)
        transported_values = character_table(transported)
        if source_values != transported_values:
            raise _validation_error(
                "result_character_binding",
                "transported character must agree with the source on every residue",
            )
        if self.values != source_values:
            raise _validation_error(
                "result_value_binding",
                "the value table must equal the complete residue table of the retained character",
            )
        return self


__all__ = [
    "DirichletCharacterBasisChangeRequest",
    "DirichletCharacterBasisChangeResult",
    "DirichletCharacterCoordinateIsomorphism",
]
