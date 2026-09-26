"""Typed contracts for exact changes of Dirichlet-character coordinates."""

from __future__ import annotations

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
        return self


__all__ = [
    "DirichletCharacterBasisChangeRequest",
    "DirichletCharacterBasisChangeResult",
    "DirichletCharacterCoordinateIsomorphism",
]
