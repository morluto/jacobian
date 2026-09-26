"""Exact Hecke matrices on multidimensional character-coordinate spaces."""

from jacobian.math.number_theory.modular_forms.multidimensional_hecke.models import (
    ModularCharacterHeckeMatrixRequest,
    ModularCharacterHeckeMatrixResult,
)
from jacobian.math.number_theory.modular_forms.multidimensional_hecke.operations import (
    modular_character_hecke_matrix_multidimensional,
)

__all__ = [
    "ModularCharacterHeckeMatrixRequest",
    "ModularCharacterHeckeMatrixResult",
    "modular_character_hecke_matrix_multidimensional",
]
