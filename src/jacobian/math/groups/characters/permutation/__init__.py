"""Character values induced by finite permutation actions."""

from jacobian.math.groups.characters.permutation._models import FiniteCharacter
from jacobian.math.groups.characters.permutation.operations import (
    permutation_character,
)

__all__ = ["FiniteCharacter", "permutation_character"]
