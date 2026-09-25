"""Exact dimensions of bounded table-bound finite-group characters."""

from jacobian.math.groups.characters.degree._models import (
    CharacterDegree,
    CharacterDegreeRequest,
)
from jacobian.math.groups.characters.degree.operations import character_degree

__all__ = ["CharacterDegree", "CharacterDegreeRequest", "character_degree"]
