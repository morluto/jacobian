"""Catalog declarations for bounded principal Dirichlet characters."""

from __future__ import annotations

from typing import Literal, cast

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.characters import operations as native
from jacobian.math.number_theory.characters._models import (
    CharacterGroupRequest,
    DirichletCharacterProductRequest,
    DirichletCharacterRequest,
    DirichletCharacterTableResult,
    DirichletCharacterValueRequest,
    DirichletCharacterValueResult,
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
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


_GROUP_MOD3 = {
    "modulus": 3,
    "unit_residues": [1, 2],
    "character_count": 2,
    "invariant_factors": [2],
    "generators": [2],
    "generator_orders": [2],
    "unit_coordinates": [[0], [1]],
    "exponent": 2,
}


def _compute_character(request: DirichletCharacterRequest) -> DirichletCharacter:
    return native.dirichlet_character(request.group, request.coordinates)


def _compute_character_value(
    request: DirichletCharacterValueRequest,
) -> DirichletCharacterValueResult:
    return native.dirichlet_character_value(request.character, int(request.integer))


def _compute_character_product(
    request: DirichletCharacterProductRequest,
) -> DirichletCharacter:
    return native.dirichlet_character_product(request.left, request.right)


def _compute_character_table(
    request: DirichletCharacterRequest,
) -> DirichletCharacterTableResult:
    return native.dirichlet_character_table(
        native.dirichlet_character(request.group, request.coordinates)
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="dirichlet_character.compute",
        title="Construct an exact Dirichlet character",
        description="Construct one character from exact dual coordinates bound to a finite unit-group parent; coordinates must fit every generator order.",
        request_type=DirichletCharacterRequest,
        result_type=DirichletCharacter,
        run=_compute_character,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="quadratic_mod3",
                description="Construct the nonprincipal character modulo 3; coordinates must use the supplied group's dual axis.",
                input={"group": _GROUP_MOD3, "coordinates": [1]},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.value.compute",
        title="Evaluate an exact Dirichlet character",
        description="Evaluate a character exactly at an integer, returning zero off the unit group and a cyclotomic root with its modulus parent on units.",
        request_type=DirichletCharacterValueRequest,
        result_type=DirichletCharacterValueResult,
        run=_compute_character_value,
        tags=("number-theory", "dirichlet-character", "cyclotomic", "exact"),
        examples=(
            OperationExample(
                name="quadratic_value_mod3",
                description="Evaluate the quadratic character modulo 3 at 2; the character must retain the exact modulus and group parent.",
                input={
                    "character": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "integer": "2",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.table.compute",
        title="Materialize an exact Dirichlet character table",
        description="Materialize the complete extension-by-zero table of one exact character, retaining modulus and cyclotomic parent identity.",
        request_type=DirichletCharacterRequest,
        result_type=DirichletCharacterTableResult,
        run=_compute_character_table,
        tags=("number-theory", "dirichlet-character", "cyclotomic", "exact"),
        examples=(
            OperationExample(
                name="table_mod3",
                description="Materialize the character table modulo 3; the character coordinates must belong to the supplied exact group.",
                input={"group": _GROUP_MOD3, "coordinates": [1]},
            ),
        ),
    ),
    MathTool(
        operation_id="dirichlet_character.multiply.compute",
        title="Multiply exact Dirichlet characters",
        description="Multiply two characters pointwise through their common finite dual-group coordinates; both operands must have the identical group parent.",
        request_type=DirichletCharacterProductRequest,
        result_type=DirichletCharacter,
        run=_compute_character_product,
        tags=("number-theory", "dirichlet-character", "exact"),
        examples=(
            OperationExample(
                name="quadratic_square_mod3",
                description="Multiply the quadratic character modulo 3 by itself; both operands must use the identical group parent.",
                input={
                    "left": {"group": _GROUP_MOD3, "coordinates": [1]},
                    "right": {"group": _GROUP_MOD3, "coordinates": [1]},
                },
            ),
        ),
    ),
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
