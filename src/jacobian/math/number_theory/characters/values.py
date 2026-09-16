"""Canonical exact values for bounded Dirichlet-character operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_PRINCIPAL_CHARACTER_MODULUS = 2_048
MAX_CHARACTER_GROUP_MODULUS = 2_048


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Dirichlet-character values."""

    return PydanticCustomError(f"dirichlet_character.{reason}", message)


class PrincipalDirichletCharacter(StrictModel):
    """The extension-by-zero principal character modulo one fixed modulus.

    ``values[a]`` is the exact value of the principal character at the
    canonical residue ``a``.  The unit residues and complete table bind this
    value to its modulus without relying on a backend-specific group basis.
    """

    modulus: StrictInt = Field(ge=1, le=MAX_PRINCIPAL_CHARACTER_MODULUS)
    unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_PRINCIPAL_CHARACTER_MODULUS
    )
    values: tuple[Literal[0, 1], ...] = Field(
        min_length=1, max_length=MAX_PRINCIPAL_CHARACTER_MODULUS
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        """Validate the bounded wire shape without proving the character."""

        if len(self.unit_residues) > self.modulus:
            raise _validation_error(
                "unit_residues_length",
                "unit residues cannot contain more entries than the modulus",
            )
        if any(
            residue < 0 or residue >= self.modulus for residue in self.unit_residues
        ):
            raise _validation_error(
                "unit_residue_range",
                "unit residues must be distinct canonical residues modulo modulus",
            )
        if self.unit_residues != tuple(sorted(set(self.unit_residues))):
            raise _validation_error(
                "unit_residue_order",
                "unit residues must be strictly increasing canonical residues",
            )
        if len(self.values) != self.modulus:
            raise _validation_error(
                "values_table_length",
                "values must contain exactly one entry for every canonical residue",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        unit_residues: tuple[StrictInt, ...],
        values: tuple[Literal[0, 1], ...],
    ) -> Self:
        """Build a character after its producer has established its table."""

        return cls.model_construct(
            modulus=modulus,
            unit_residues=unit_residues,
            values=values,
        )


class DirichletCharacterGroup(StrictModel):
    """The finite unit group modulo one fixed modulus with character coordinates.

    ``unit_residues`` lists every unit in increasing canonical order.
    ``generators`` is a tuple of canonical unit residues whose orders are
    ``generator_orders``; every unit's ``unit_coordinates`` row holds the
    discrete logarithms of that unit against those generators, in generator
    order. ``invariant_factors`` is the divisibility chain of the finite
    Abelian unit group. ``exponent`` is the common root-of-unity order of the
    dual character group, and ``character_count`` equals ``phi(modulus)``.
    """

    modulus: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    character_count: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    invariant_factors: tuple[StrictInt, ...] = Field(
        max_length=32,
        description=(
            "Divisibility chain of the finite Abelian unit group; empty exactly "
            "for the trivial group."
        ),
    )
    generators: tuple[StrictInt, ...] = Field(
        max_length=32,
        description="Canonical unit residues generating the unit group.",
    )
    generator_orders: tuple[StrictInt, ...] = Field(max_length=32)
    unit_coordinates: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Discrete-logarithm rows aligned with unit_residues, in generator "
            "order; each entry is below the corresponding generator order."
        ),
    )
    exponent: StrictInt = Field(
        ge=1,
        description="Common root-of-unity order of the dual character group.",
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:  # noqa: C901
        """Validate the bounded wire shape without proving the group."""

        if len(self.unit_residues) > self.modulus:
            raise _validation_error(
                "unit_residues_length",
                "unit residues cannot contain more entries than the modulus",
            )
        if any(
            residue < 0 or residue >= self.modulus for residue in self.unit_residues
        ):
            raise _validation_error(
                "unit_residue_range",
                "unit residues must be distinct canonical residues modulo modulus",
            )
        if self.unit_residues != tuple(sorted(set(self.unit_residues))):
            raise _validation_error(
                "unit_residue_order",
                "unit residues must be strictly increasing canonical residues",
            )
        if self.character_count != len(self.unit_residues):
            raise _validation_error(
                "character_count_mismatch",
                "character count must equal the number of unit residues",
            )
        if len(self.generators) != len(self.generator_orders):
            raise _validation_error(
                "generator_order_length",
                "generators and generator orders must align",
            )
        if any(
            generator < 0 or generator >= self.modulus for generator in self.generators
        ):
            raise _validation_error(
                "generator_range",
                "generators must be canonical residues modulo modulus",
            )
        if any(order < 1 for order in self.generator_orders):
            raise _validation_error(
                "generator_order_range",
                "generator orders must be positive",
            )
        if len(self.unit_coordinates) != len(self.unit_residues):
            raise _validation_error(
                "coordinate_row_count",
                "coordinate rows must align with unit residues",
            )
        for row in self.unit_coordinates:
            if len(row) != len(self.generators):
                raise _validation_error(
                    "coordinate_row_length",
                    "coordinate rows must align with generators",
                )
            for coordinate, order in zip(row, self.generator_orders, strict=True):
                if coordinate < 0 or coordinate >= order:
                    raise _validation_error(
                        "coordinate_range",
                        "coordinates must lie below their generator order",
                    )
        if self.invariant_factors != tuple(sorted(self.invariant_factors)):
            raise _validation_error(
                "invariant_factor_order",
                "invariant factors must be sorted increasingly",
            )
        for first, second in zip(
            self.invariant_factors, self.invariant_factors[1:], strict=False
        ):
            if second % first != 0:
                raise _validation_error(
                    "invariant_factor_divisibility",
                    "each invariant factor must divide the next",
                )
        if self.exponent < 1:
            raise _validation_error(
                "exponent_range",
                "the common root-of-unity exponent must be positive",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        unit_residues: tuple[int, ...],
        character_count: int,
        invariant_factors: tuple[int, ...],
        generators: tuple[int, ...],
        generator_orders: tuple[int, ...],
        unit_coordinates: tuple[tuple[int, ...], ...],
        exponent: int,
    ) -> Self:
        """Build a group after its producer has established the decomposition."""

        return cls.model_construct(
            modulus=modulus,
            unit_residues=unit_residues,
            character_count=character_count,
            invariant_factors=invariant_factors,
            generators=generators,
            generator_orders=generator_orders,
            unit_coordinates=unit_coordinates,
            exponent=exponent,
        )


__all__ = [
    "MAX_CHARACTER_GROUP_MODULUS",
    "MAX_PRINCIPAL_CHARACTER_MODULUS",
    "DirichletCharacterGroup",
    "PrincipalDirichletCharacter",
]
