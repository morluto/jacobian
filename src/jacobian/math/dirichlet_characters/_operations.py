"""MCP-facing bounded principal Dirichlet-character kernels."""

from __future__ import annotations

from typing import Literal, cast

from jacobian.math.dirichlet_characters._models import (
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.dirichlet_characters.operations import (
    principal_dirichlet_character,
    principal_dirichlet_character_value,
)
from jacobian.math.dirichlet_characters.values import PrincipalDirichletCharacter


def compute_principal_dirichlet_character(
    request: PrincipalDirichletCharacterRequest,
) -> PrincipalDirichletCharacter:
    """Materialize the complete exact principal character for one modulus."""

    return principal_dirichlet_character(request.modulus)


def compute_principal_dirichlet_character_value(
    request: PrincipalDirichletCharacterValueRequest,
) -> PrincipalDirichletCharacterValueResult:
    """Evaluate one principal character at a source-bound exact integer."""

    value = principal_dirichlet_character_value(
        request.character, request.integer_value()
    )
    residue = request.integer_value() % request.character.modulus
    return PrincipalDirichletCharacterValueResult(
        character=request.character,
        integer=request.integer,
        canonical_residue=residue,
        is_unit=value == 1,
        value=cast(Literal[0, 1], value),
    )


__all__ = [
    "compute_principal_dirichlet_character",
    "compute_principal_dirichlet_character_value",
]
