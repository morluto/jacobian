"""Known-answer, boundary, adversarial, and invariant tests for character groups."""

from __future__ import annotations

import json
from math import gcd, prod

import pytest
from pydantic import ValidationError
from sympy import totient

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters import (
    character_group,
    require_complete_character_group,
)
from jacobian.math.number_theory.characters._models import CharacterGroupRequest
from jacobian.math.number_theory.characters._tools import (
    TOOLS,
    compute_character_group,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    DirichletCharacterGroup,
)


def _rebuild(group: DirichletCharacterGroup, residue: int) -> int:
    row = group.unit_coordinates[group.unit_residues.index(residue)]
    rebuilt = 1 % group.modulus
    for generator, coordinate in zip(group.generators, row, strict=True):
        rebuilt = (rebuilt * pow(generator, coordinate, group.modulus)) % group.modulus
    return rebuilt


def test_group_modulo_twelve_has_canonical_decomposition() -> None:
    group = compute_character_group(CharacterGroupRequest(modulus=12))

    assert group.unit_residues == (1, 5, 7, 11)
    assert group.character_count == 4
    assert group.invariant_factors == (2, 2)
    assert group.generator_orders == (2, 2)
    assert all(gcd(generator, 12) == 1 for generator in group.generators)
    assert group.exponent == 2


def test_group_modulo_eight_uses_two_generator_presentation() -> None:
    group = character_group(8)

    assert group.unit_residues == (1, 3, 5, 7)
    assert group.invariant_factors == (2, 2)
    assert group.exponent == 2
    assert group.character_count == 4


def test_cyclic_prime_power_group_has_single_generator() -> None:
    group = character_group(7)

    assert group.invariant_factors == (6,)
    assert group.generators == (3,)
    assert group.generator_orders == (6,)
    assert group.exponent == 6


@pytest.mark.parametrize("modulus", [1, 2, 3, 4, 5, 8, 12])
def test_small_moduli_conventions(modulus: int) -> None:
    group = character_group(modulus)

    assert group.character_count == int(totient(modulus))
    assert group.character_count == len(group.unit_residues)
    assert prod(group.invariant_factors, start=1) == group.character_count


def test_degenerate_moduli_have_trivial_presentations() -> None:
    trivial = character_group(1)

    assert trivial.unit_residues == (0,)
    assert trivial.character_count == 1
    assert trivial.invariant_factors == ()
    assert trivial.generators == ()
    assert trivial.generator_orders == ()
    assert trivial.unit_coordinates == ((),)
    assert trivial.exponent == 1

    two = character_group(2)
    assert two.unit_residues == (1,)
    assert two.invariant_factors == ()
    assert two.generators == ()


@pytest.mark.parametrize("modulus", [9, 15, 16, 24, 25, 27, 36, 48, 49, 72, 100])
def test_generator_coordinate_round_trip_identity_on_units(modulus: int) -> None:
    group = character_group(modulus)
    require_complete_character_group(group)

    assert group.character_count == int(totient(modulus))
    for residue in group.unit_residues:
        assert _rebuild(group, residue) == residue


def test_native_and_catalog_paths_agree() -> None:
    native = character_group(12)
    catalog = compute_character_group(CharacterGroupRequest(modulus=12))

    assert catalog == native
    assert catalog.model_dump(mode="json") == native.model_dump(mode="json")


def test_group_value_round_trips_through_json() -> None:
    group = character_group(15)
    assert DirichletCharacterGroup.model_validate_json(group.model_dump_json()) == group


def test_group_checker_rejects_a_forged_coordinate_claim() -> None:
    group = character_group(12)
    payload = group.model_dump(mode="json")
    payload["unit_coordinates"][0] = [1, 0]
    forged = DirichletCharacterGroup.model_validate_json(json.dumps(payload))

    with pytest.raises(OperationDomainValidationError) as excinfo:
        require_complete_character_group(forged)
    assert excinfo.value.errors()[0]["type"].startswith("dirichlet_character.group")


def test_group_checker_rejects_a_swapped_generator_claim() -> None:
    group = character_group(7)
    payload = group.model_dump(mode="json")
    payload["generators"] = [5]
    forged = DirichletCharacterGroup.model_validate_json(json.dumps(payload))

    with pytest.raises(OperationDomainValidationError) as excinfo:
        require_complete_character_group(forged)
    assert excinfo.value.errors()[0]["type"].startswith("dirichlet_character.group")


def test_group_model_rejects_structurally_invalid_shapes() -> None:
    group = character_group(12)
    payload = group.model_dump(mode="json")
    payload["invariant_factors"] = [3, 2]
    with pytest.raises(ValidationError) as error:
        DirichletCharacterGroup.model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.invariant_factor_order"
    )

    payload = group.model_dump(mode="json")
    payload["unit_coordinates"][1] = [2, 0]
    with pytest.raises(ValidationError) as error:
        DirichletCharacterGroup.model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == "dirichlet_character.coordinate_range"


def test_modulus_boundary_is_complete_and_next_value_is_rejected() -> None:
    group = character_group(MAX_CHARACTER_GROUP_MODULUS)

    assert len(group.unit_residues) == int(totient(MAX_CHARACTER_GROUP_MODULUS))
    with pytest.raises(ValidationError):
        CharacterGroupRequest(modulus=MAX_CHARACTER_GROUP_MODULUS + 1)
    with pytest.raises(OperationResourceAdmissionError):
        character_group(MAX_CHARACTER_GROUP_MODULUS + 1)


def test_native_admission_rejects_nonpositive_and_boolean_moduli() -> None:
    with pytest.raises(OperationDomainValidationError):
        character_group(0)
    with pytest.raises(OperationDomainValidationError):
        character_group(True)


def test_catalog_declares_the_group_operation_with_example() -> None:
    operation_ids = tuple(tool.operation_id for tool in TOOLS)

    assert "dirichlet_character.group.compute" in operation_ids
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "dirichlet_character.group.compute"
    )
    assert tool.examples
    assert len(tool.discovery_terms) <= 8
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    tool.result_type.model_validate_json(json.dumps(result.model_dump(mode="json")))
