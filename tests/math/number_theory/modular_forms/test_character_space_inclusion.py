"""Exact nested-level inclusion for character-valued modular spaces."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters.operations import character_group
from jacobian.math.number_theory.characters.values import DirichletCharacter
from jacobian.math.number_theory.modular_forms._models import (
    ModularCharacterSpaceInclusionRequest,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.space_maps import (
    modular_form_character_space_inclusion,
    require_modular_character_space_inclusion,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_MODULAR_CHARACTER_INCLUSION_LEVEL,
    ModularCharacterSpaceInclusion,
    ModularFormSpace,
)


def _character(group, coordinates: tuple[int, ...]) -> DirichletCharacter:
    return DirichletCharacter(group=group, coordinates=coordinates)


def _angle(character: DirichletCharacter, integer: int) -> Fraction | None:
    """Evaluate from the finite-unit decomposition, independently of the map."""

    group = character.group
    residue = integer % group.modulus
    try:
        row = group.unit_coordinates[group.unit_residues.index(residue)]
    except ValueError:
        return None
    exponent = sum(
        coordinate * (group.exponent // order) * unit_coordinate
        for coordinate, order, unit_coordinate in zip(
            character.coordinates, group.generator_orders, row, strict=True
        )
    )
    return Fraction(exponent % group.exponent, group.exponent)


def _inflated_character(source: DirichletCharacter, target_modulus: int):
    """Find the exact target dual coordinates by checking every target unit."""

    target_group = character_group(target_modulus)
    axes = tuple(range(order) for order in target_group.generator_orders)
    for coordinates in product(*axes):
        candidate = _character(target_group, coordinates)
        if all(
            _angle(candidate, residue) == _angle(source, residue)
            for residue in target_group.unit_residues
        ):
            return candidate
    raise AssertionError("the unit-reduction pullback must have target coordinates")


def _space(level: int, character: DirichletCharacter) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind="S",
        character=character,
        coefficient_domain="QQ",
    )


def test_quadratic_character_inclusion_matches_every_target_unit() -> None:
    source_character = _character(character_group(3), (1,))
    target_character = _inflated_character(source_character, 15)

    inclusion = modular_form_character_space_inclusion(
        _space(3, source_character), _space(15, target_character)
    )

    assert inclusion.map_kind == "gamma0_character_inflation"
    assert inclusion.source_space.level == 3
    assert inclusion.target_space.level == 15
    assert inclusion.source_space.weight == inclusion.target_space.weight
    for residue in range(15):
        if residue in target_character.group.unit_residues:
            assert _angle(target_character, residue) == _angle(
                source_character, residue % 3
            )
        else:
            assert _angle(target_character, residue) is None


def test_character_inclusion_is_parented_and_survives_serialization() -> None:
    source = _character(character_group(3), (1,))
    target = _inflated_character(source, 15)
    value = modular_form_character_space_inclusion(
        _space(3, source), _space(15, target)
    )

    restored = ModularCharacterSpaceInclusion.model_validate_json(
        value.model_dump_json()
    )

    assert require_modular_character_space_inclusion(restored) == value


def test_character_inclusion_rejects_noninflated_target_character() -> None:
    source = _character(character_group(3), (1,))
    wrong_target = _character(character_group(15), (0, 0))

    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_character_space_inclusion(
            _space(3, source), _space(15, wrong_target)
        )

    assert (
        error.value.errors()[0]["type"] == "modular_form.character_inflation_mismatch"
    )


def test_consumer_rechecks_serialized_map_claims() -> None:
    source = _character(character_group(3), (1,))
    target = _character(character_group(15), (0, 0))
    forged = ModularCharacterSpaceInclusion.model_construct(
        map_kind="gamma0_character_inflation",
        source_space=_space(3, source),
        target_space=_space(15, target),
    )

    with pytest.raises(OperationDomainValidationError):
        require_modular_character_space_inclusion(forged)


def test_character_space_inclusion_contract_and_structural_bound() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "modular_form.character_space.inclusion.compute"
    )
    source = _character(character_group(3), (1,))
    target = _inflated_character(source, 15)
    result = tool.run(
        ModularCharacterSpaceInclusionRequest(
            source_space=_space(3, source), target_space=_space(15, target)
        )
    )
    assert isinstance(result, ModularCharacterSpaceInclusion)


def test_character_inclusion_rejects_constructed_structurally_invalid_map() -> None:
    source_character = _character(character_group(3), (0,))
    target_character = _character(character_group(3), (0,))
    forged = ModularCharacterSpaceInclusion.model_construct(
        map_kind="gamma0_character_inflation",
        source_space=_space(3, source_character),
        target_space=ModularFormSpace(
            level=3,
            weight=4,
            kind="S",
            character=target_character,
            coefficient_domain="QQ",
        ),
    )

    with pytest.raises(OperationDomainValidationError):
        require_modular_character_space_inclusion(forged)


def test_character_inclusion_rejects_incomplete_authored_group() -> None:
    canonical_group = character_group(3)
    incomplete_group = type(canonical_group).model_construct(
        **{
            **canonical_group.model_dump(),
            "unit_residues": (1,),
            "unit_coordinates": ((0,),),
            "character_count": 1,
        }
    )
    fabricated = DirichletCharacter.model_construct(
        group=incomplete_group, coordinates=(0,)
    )
    forged_space = ModularFormSpace.model_construct(
        group="GAMMA0",
        level=3,
        weight=2,
        kind="S",
        character=fabricated,
        coefficient_domain="QQ",
    )

    with pytest.raises(OperationDomainValidationError):
        modular_form_character_space_inclusion(forged_space, forged_space)


def test_character_inclusion_accepts_maximum_admitted_level() -> None:
    level = MAX_MODULAR_CHARACTER_INCLUSION_LEVEL
    group = character_group(level)
    principal = _character(group, (0,) * len(group.generator_orders))

    inclusion = modular_form_character_space_inclusion(
        _space(level, principal), _space(level, principal)
    )

    assert inclusion.source_space.level == level
    assert inclusion.target_space.level == level
    assert len(group.unit_residues) == group.character_count
