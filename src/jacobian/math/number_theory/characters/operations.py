"""Native exact operations for bounded principal Dirichlet characters."""

from __future__ import annotations

import math

from jacobian.math.number_theory.characters._models import (
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.number_theory.characters.values import (
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    PrincipalDirichletCharacter,
)

__all__ = [
    "principal_dirichlet_character",
    "principal_dirichlet_character_value",
    "require_complete_principal_dirichlet_character",
    "require_principal_dirichlet_character_value_result",
]


def require_complete_principal_dirichlet_character(
    character: PrincipalDirichletCharacter,
) -> None:
    """Check the mathematical table claimed by a caller-supplied character."""

    expected_units = tuple(
        residue
        for residue in range(character.modulus)
        if math.gcd(residue, character.modulus) == 1
    )
    if character.unit_residues != expected_units:
        raise ValueError(
            "unit residues must be the complete canonical unit group modulo modulus"
        )
    units = frozenset(expected_units)
    expected_values = tuple(
        1 if residue in units else 0 for residue in range(character.modulus)
    )
    if character.values != expected_values:
        raise ValueError(
            "values must be the complete extension-by-zero principal character table"
        )


def require_principal_dirichlet_character_value_result(
    result: PrincipalDirichletCharacterValueResult,
) -> None:
    """Check the character and source-evaluation relation of a claimed result."""

    require_complete_principal_dirichlet_character(result.character)
    residue = int(result.integer) % result.character.modulus
    if result.canonical_residue != residue:
        raise ValueError("canonical residue does not match the source integer")
    expected_is_unit = math.gcd(residue, result.character.modulus) == 1
    if result.is_unit != expected_is_unit:
        raise ValueError("unit status does not match the source character modulus")
    if result.value != result.character.values[residue]:
        raise ValueError("value does not match the source principal-character table")


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
    return PrincipalDirichletCharacter._from_kernel(
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
    require_complete_principal_dirichlet_character(character)
    return character.values[integer % character.modulus]
