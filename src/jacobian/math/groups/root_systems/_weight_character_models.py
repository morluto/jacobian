"""Typed finite highest-weight characters for type A."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.groups.root_systems._cartan import cartan_type_matrix
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    CartanMatrix,
    CartanMatrixRequest,
    _validation_error,
)

MAX_CHARACTER_STATES = 4_096
MAX_CHARACTER_MULTIPLICITY_BITS = 16_384


class HighestWeightCharacterRequest(CartanMatrixRequest):
    """A dominant highest weight in fundamental-weight coordinates."""

    highest_weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)

    @model_validator(mode="after")
    def require_dominant_weight_axis(self) -> Self:
        if len(self.highest_weight) != len(self.matrix) or any(
            value < 0 for value in self.highest_weight
        ):
            raise _validation_error(
                "invalid_dominant_weight",
                "highest weight must be nonnegative and match Cartan rank",
            )
        return self


class WeightMultiplicity(StrictModel):
    """One weight and its positive multiplicity in an irreducible module."""

    weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    multiplicity: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def require_bounded_coordinates_and_multiplicity(self) -> Self:
        if (
            any(value.bit_length() > MAX_CHARACTER_STATES.bit_length() for value in self.weight)
            or self.multiplicity.bit_length() > MAX_CHARACTER_MULTIPLICITY_BITS
        ):
            raise PydanticCustomError(
                "root_system.character_term_bounds",
                "character weight coordinates and multiplicity must be bounded",
            )
        return self


class IrreducibleWeightCharacter(StrictModel):
    """A complete, canonical finite weight-multiplicity table."""

    matrix: CartanMatrix
    weight_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_RANK)
    highest_weight: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    terms: tuple[WeightMultiplicity, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_STATES
    )

    @model_validator(mode="after")
    def require_canonical_character(self) -> Self:
        rank = len(self.matrix)
        weights = tuple(term.weight for term in self.terms)
        if (
            self.matrix.entries != cartan_type_matrix("A", rank)
            or
            self.weight_axis != tuple(range(rank))
            or len(self.highest_weight) != rank
            or any(value < 0 for value in self.highest_weight)
            or any(
                value.bit_length() > MAX_CHARACTER_STATES.bit_length()
                for value in self.highest_weight
            )
            or any(len(weight) != rank for weight in weights)
            or weights != tuple(sorted(set(weights)))
            or self.highest_weight not in weights
            or self.terms[weights.index(self.highest_weight)].multiplicity != 1
        ):
            raise _validation_error(
                "character_shape",
                "character terms must be unique and sorted on the Cartan weight axis, "
                "with highest weight of multiplicity one"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matrix: CartanMatrix,
        highest_weight: tuple[int, ...],
        terms: tuple[WeightMultiplicity, ...],
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            weight_axis=tuple(range(len(matrix))),
            highest_weight=highest_weight,
            terms=terms,
        )
