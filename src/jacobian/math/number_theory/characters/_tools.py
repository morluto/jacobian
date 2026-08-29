"""Catalog declarations for bounded principal Dirichlet characters."""

from __future__ import annotations

from typing import Literal, cast

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.number_theory.characters import operations as native
from jacobian.math.number_theory.characters._models import (
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.number_theory.characters.values import PrincipalDirichletCharacter


def compute_principal_dirichlet_character(
    request: PrincipalDirichletCharacterRequest,
) -> PrincipalDirichletCharacter:
    """Materialize the complete exact principal character for one modulus."""

    return native.principal_dirichlet_character(request.modulus)


def compute_principal_dirichlet_character_value(
    request: PrincipalDirichletCharacterValueRequest,
) -> PrincipalDirichletCharacterValueResult:
    """Evaluate one principal Dirichlet character at a source-bound integer."""

    value = native.principal_dirichlet_character_value(
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


TOOLS: MathTools = (
    MathTool(
        operation_id="dirichlet_character.principal.compute",
        title="Compute an exact principal Dirichlet character",
        description=(
            "Materialize the complete extension-by-zero table of the principal "
            "Dirichlet character modulo a bounded positive modulus. The returned "
            "canonical value composes directly with exact character evaluation."
        ),
        request_type=PrincipalDirichletCharacterRequest,
        result_type=PrincipalDirichletCharacter,
        run=compute_principal_dirichlet_character,
        tags=("number-theory", "dirichlet-character", "principal", "exact"),
        examples=(
            example(
                "principal_character_mod_12",
                "Compute the complete principal character modulo 12; the modulus must be positive and its full residue table must fit the 2,048-entry bound.",
                {"modulus": 12},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.principal.value.compute",
        title="Evaluate a principal Dirichlet character",
        description=(
            "Evaluate the exact principal Dirichlet character at an integer, "
            "retaining its source character and canonical residue."
        ),
        request_type=PrincipalDirichletCharacterValueRequest,
        result_type=PrincipalDirichletCharacterValueResult,
        run=compute_principal_dirichlet_character_value,
        tags=("number-theory", "dirichlet-character", "principal", "exact"),
        examples=(
            example(
                "principal_character_value_mod_12",
                "Evaluate the principal character modulo 12 at 5.",
                {
                    "character": {
                        "modulus": 12,
                        "unit_residues": [1, 5, 7, 11],
                        "values": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                    },
                    "integer": "5",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
