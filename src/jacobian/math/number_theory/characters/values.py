"""Canonical exact values for bounded Dirichlet-character operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_PRINCIPAL_CHARACTER_MODULUS = 2_048


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


__all__ = ["MAX_PRINCIPAL_CHARACTER_MODULUS", "PrincipalDirichletCharacter"]
