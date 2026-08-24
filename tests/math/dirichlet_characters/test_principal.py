"""Known-answer and source-binding tests for principal Dirichlet characters."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.dirichlet_characters import (
    principal_dirichlet_character,
    principal_dirichlet_character_value,
)
from jacobian.math.dirichlet_characters._models import (
    PrincipalDirichletCharacterRequest,
    PrincipalDirichletCharacterValueRequest,
    PrincipalDirichletCharacterValueResult,
)
from jacobian.math.dirichlet_characters._operations import (
    compute_principal_dirichlet_character,
    compute_principal_dirichlet_character_value,
)
from jacobian.math.dirichlet_characters._tools import TOOLS
from jacobian.math.dirichlet_characters.values import (
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    PrincipalDirichletCharacter,
)


def test_principal_character_modulo_twelve_has_complete_extension_by_zero_table() -> (
    None
):
    character = compute_principal_dirichlet_character(
        PrincipalDirichletCharacterRequest(modulus=12)
    )

    assert character.unit_residues == (1, 5, 7, 11)
    assert character.values == (0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1)


def test_modulus_one_has_its_single_residue_as_a_unit() -> None:
    character = principal_dirichlet_character(1)

    assert character.unit_residues == (0,)
    assert character.values == (1,)
    assert principal_dirichlet_character_value(character, -9) == 1


@pytest.mark.parametrize(
    ("integer", "residue", "is_unit", "value"),
    [
        ("25", 1, True, 1),
        ("-1", 11, True, 1),
        ("18", 6, False, 0),
    ],
)
def test_value_operation_is_bound_to_the_supplied_table(
    integer: str, residue: int, is_unit: bool, value: int
) -> None:
    character = principal_dirichlet_character(12)
    result = compute_principal_dirichlet_character_value(
        PrincipalDirichletCharacterValueRequest(character=character, integer=integer)
    )

    assert result.character == character
    assert result.canonical_residue == residue
    assert result.is_unit is is_unit
    assert result.value == value


def test_character_model_rejects_forged_unit_group_or_value_table() -> None:
    with pytest.raises(ValidationError, match="unit residues"):
        PrincipalDirichletCharacter(
            modulus=12,
            unit_residues=(1, 5, 7),
            values=(0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1),
        )
    with pytest.raises(ValidationError, match="extension-by-zero"):
        PrincipalDirichletCharacter(
            modulus=12,
            unit_residues=(1, 5, 7, 11),
            values=(0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1),
        )


def test_value_result_rejects_a_forged_residue_or_value() -> None:
    character = principal_dirichlet_character(12)
    with pytest.raises(ValidationError, match="canonical residue"):
        PrincipalDirichletCharacterValueResult(
            character=character,
            integer="25",
            canonical_residue=5,
            is_unit=True,
            value=1,
        )
    with pytest.raises(ValidationError, match="value does not match"):
        PrincipalDirichletCharacterValueResult(
            character=character,
            integer="18",
            canonical_residue=6,
            is_unit=False,
            value=1,
        )


def test_value_request_rejects_noncanonical_negative_zero() -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        PrincipalDirichletCharacterValueRequest(
            character=principal_dirichlet_character(12), integer="-0"
        )


def test_modulus_boundary_is_complete_and_next_value_is_rejected() -> None:
    character = principal_dirichlet_character(MAX_PRINCIPAL_CHARACTER_MODULUS)

    assert len(character.values) == MAX_PRINCIPAL_CHARACTER_MODULUS
    with pytest.raises(ValidationError, match="less than or equal"):
        PrincipalDirichletCharacterRequest(modulus=MAX_PRINCIPAL_CHARACTER_MODULUS + 1)


def test_native_api_rejects_boolean_modulus_and_integer() -> None:
    with pytest.raises(TypeError, match="modulus"):
        principal_dirichlet_character(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="input"):
        principal_dirichlet_character_value(principal_dirichlet_character(3), True)  # type: ignore[arg-type]


def test_catalog_declares_the_composable_principal_operations() -> None:
    assert tuple(tool.operation_id for tool in TOOLS) == (
        "dirichlet_character.principal.compute",
        "dirichlet_character.principal.value.compute",
    )
    assert all(tool.version == "1" for tool in TOOLS)
