"""Canonical exact values for bounded Dirichlet-character operations."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel

MAX_PRINCIPAL_CHARACTER_MODULUS = 2_048


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
    def require_complete_canonical_principal_table(self) -> Self:
        expected_units = tuple(
            residue
            for residue in range(self.modulus)
            if math.gcd(residue, self.modulus) == 1
        )
        if self.unit_residues != expected_units:
            raise ValueError(
                "unit residues must be the complete canonical unit group modulo modulus"
            )
        units = frozenset(expected_units)
        expected_values = tuple(
            1 if residue in units else 0 for residue in range(self.modulus)
        )
        if self.values != expected_values:
            raise ValueError(
                "values must be the complete extension-by-zero principal character table"
            )
        return self


__all__ = ["MAX_PRINCIPAL_CHARACTER_MODULUS", "PrincipalDirichletCharacter"]
