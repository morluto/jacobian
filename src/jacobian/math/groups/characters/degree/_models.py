"""Contracts for exact degrees of table-bound ordinary characters."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.groups.characters._models import CharacterRingElement


class CharacterDegreeRequest(StrictModel):
    """An ordinary character in the irreducible basis of a finite-group table."""

    character: CharacterRingElement


class CharacterDegree(StrictModel):
    """The exact dimension of an ordinary character, retaining its source basis."""

    character: CharacterRingElement
    degree: ExactInteger

    @model_validator(mode="after")
    def require_degree_relation(self) -> Self:
        multiplicities = self.character.irreducible_multiplicities
        rows = self.character.table.rows
        if len(multiplicities) != len(rows) or any(
            value < 0 for value in multiplicities
        ):
            raise ValueError("character degree requires complete ordinary coordinates")
        if self.degree != sum(
            multiplicity * row.degree
            for multiplicity, row in zip(multiplicities, rows, strict=True)
        ):
            raise ValueError("character degree must equal the identity-class value")
        return self


__all__ = ["CharacterDegree", "CharacterDegreeRequest"]
