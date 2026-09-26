"""Exact character inflation along reduction of finite unit groups."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_inflate,
)
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterInflationRequest,
)
from jacobian.math.number_theory.characters._tools import TOOLS
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character,
    dirichlet_character_value,
)


def test_inflation_matches_reduction_on_units_and_is_zero_on_target_nonunits():
    source = dirichlet_character(character_group(3), (1,))

    result = dirichlet_character_inflate(source, 15)

    assert result.source == source
    assert result.target.group.modulus == 15
    assert result.target_unit_residues == character_group(15).unit_residues
    assert result.source_unit_residues == tuple(
        residue % 3 for residue in result.target_unit_residues
    )
    source_exponent = source.group.exponent
    target_exponent = result.target.group.exponent
    assert target_exponent % source_exponent == 0
    root_embedding = target_exponent // source_exponent
    for residue in range(15):
        source_value = dirichlet_character_value(source, residue).value
        target_value = dirichlet_character_value(result.target, residue).value
        if residue in result.target_unit_residues:
            assert source_value is not None and target_value is not None
            assert target_value.exponent == source_value.exponent * root_embedding
        else:
            assert target_value is None


def test_inflation_is_identity_at_same_modulus_and_handles_modulus_one():
    source = dirichlet_character(character_group(5), (1,))
    identity = dirichlet_character_inflate(source, 5)
    assert identity.target == source
    assert identity.source_unit_residues == identity.target_unit_residues

    trivial_modulus_one = dirichlet_character(character_group(1), ())
    inflated = dirichlet_character_inflate(trivial_modulus_one, 8)
    assert inflated.target == dirichlet_character(character_group(8), (0, 0))
    assert all(
        dirichlet_character_value(inflated.target, residue).value is not None
        for residue in inflated.target_unit_residues
    )
    assert dirichlet_character_value(inflated.target, 2).value is None


def test_inflation_accepts_the_maximum_target_group_with_bounded_map():
    trivial_modulus_one = dirichlet_character(character_group(1), ())

    result = dirichlet_character_inflate(trivial_modulus_one, 2048)

    assert result.target.group.modulus == 2048
    assert len(result.target_unit_residues) == result.target.group.character_count


def test_inflation_rejects_a_target_not_divisible_by_source_modulus():
    source = dirichlet_character(character_group(3), (1,))

    with pytest.raises(OperationDomainValidationError) as error:
        dirichlet_character_inflate(source, 10)

    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.inflation.modulus_not_multiple"
    )

    with pytest.raises(OperationResourceAdmissionError):
        dirichlet_character_inflate(dirichlet_character(character_group(1), ()), 2049)


def test_inflation_request_and_catalog_contract_are_published():
    with pytest.raises(ValidationError):
        DirichletCharacterInflationRequest(
            character=dirichlet_character(character_group(3), (1,)),
            target_modulus=4096,
        )

    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "dirichlet_character.inflate.compute"
    )
    result = tool.run(
        DirichletCharacterInflationRequest(
            character=dirichlet_character(character_group(3), (1,)),
            target_modulus=15,
        )
    )
    assert result.target.group.modulus == 15
