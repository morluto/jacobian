"""Catalog declarations for bounded principal Dirichlet characters."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.number_theory.characters._models import (
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.number_theory.characters._operations import (
    compute_principal_dirichlet_character,
    compute_principal_dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import PrincipalDirichletCharacter

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
        title="Evaluate an exact principal Dirichlet character",
        description=(
            "Evaluate a canonical principal Dirichlet character at one exact "
            "integer, returning its canonical residue, unit status, and exact "
            "extension-by-zero value bound to the supplied character."
        ),
        request_type=PrincipalDirichletCharacterValueRequest,
        result_type=PrincipalDirichletCharacterValueResult,
        run=compute_principal_dirichlet_character_value,
        tags=("number-theory", "dirichlet-character", "principal", "exact"),
        examples=(
            example(
                "principal_character_mod_12_at_25",
                "Evaluate the principal character modulo 12 at 25; the character table must be canonical and the integer must use canonical base-10 syntax.",
                {
                    "character": {
                        "modulus": 12,
                        "unit_residues": [1, 5, 7, 11],
                        "values": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                    },
                    "integer": "25",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
