"""Catalog declarations for bounded principal Dirichlet characters."""

from __future__ import annotations

from typing import Literal, cast

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.characters import operations as native
from jacobian.math.number_theory.characters._models import (
    CharacterGroupRequest,
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacterGroup,
    PrincipalDirichletCharacter,
)


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
    return PrincipalDirichletCharacterValueResult._from_kernel(
        character=request.character,
        integer=request.integer,
        canonical_residue=residue,
        is_unit=value == 1,
        value=cast(Literal[0, 1], value),
    )


def compute_character_group(request: CharacterGroupRequest) -> DirichletCharacterGroup:
    """Compute the finite unit-group decomposition for one modulus."""

    return native.character_group(request.modulus)


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
            OperationExample(
                name="principal_character_mod_12",
                description="Compute the complete principal character modulo 12; the modulus must be positive and its full residue table must fit the 2,048-entry bound.",
                input={"modulus": 12},
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
            OperationExample(
                name="principal_character_value_mod_12",
                description="Evaluate the principal character modulo 12 at 5.",
                input={
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
    MathTool(
        operation_id="dirichlet_character.group.compute",
        title="Compute a finite Dirichlet character group",
        description=(
            "Compute the finite unit group modulo a bounded positive modulus: "
            "the unit set, invariant factors, canonical generators with "
            "residue-coordinate maps, the common root-of-unity exponent, and "
            "the character count equal to phi(modulus)."
        ),
        request_type=CharacterGroupRequest,
        result_type=DirichletCharacterGroup,
        run=compute_character_group,
        tags=("number-theory", "dirichlet-character", "unit-group", "exact"),
        discovery_terms=(
            "Dirichlet character group",
            "unit group decomposition",
        ),
        examples=(
            OperationExample(
                name="character_group_mod_12",
                description="Compute the unit-group decomposition modulo 12; the modulus must be positive and its unit table must fit the 2,048-entry bound.",
                input={"modulus": 12},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
