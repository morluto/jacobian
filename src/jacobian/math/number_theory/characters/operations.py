"""Native exact operations for bounded principal Dirichlet characters."""

from __future__ import annotations

import math

from jacobian.math.number_theory.characters.values import (
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    PrincipalDirichletCharacter,
)

__all__ = ["principal_dirichlet_character", "principal_dirichlet_character_value"]


def principal_dirichlet_character(modulus: int) -> PrincipalDirichletCharacter:
    """Return the complete extension-by-zero principal character modulo ``modulus``."""

    if type(modulus) is not int:
        raise TypeError("principal-character modulus must be an integer")
    if not 1 <= modulus <= MAX_PRINCIPAL_CHARACTER_MODULUS:
        raise ValueError(
            "principal-character modulus must be between 1 and "
            f"{MAX_PRINCIPAL_CHARACTER_MODULUS}"
        )
    unit_residues = tuple(
        residue for residue in range(modulus) if math.gcd(residue, modulus) == 1
    )
    units = frozenset(unit_residues)
    return PrincipalDirichletCharacter(
        modulus=modulus,
        unit_residues=unit_residues,
        values=tuple(1 if residue in units else 0 for residue in range(modulus)),
    )


def principal_dirichlet_character_value(
    character: PrincipalDirichletCharacter, integer: int
) -> int:
    """Return the exact value of ``character`` at one integer."""

    if type(integer) is not int:
        raise TypeError("principal-character input must be an integer")
    return character.values[integer % character.modulus]
