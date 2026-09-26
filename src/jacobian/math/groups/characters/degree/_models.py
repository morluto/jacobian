"""Contracts for exact degrees of table-bound ordinary characters."""

from __future__ import annotations

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


__all__ = ["CharacterDegree", "CharacterDegreeRequest"]
