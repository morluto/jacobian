"""Exact conductor computation for finite Dirichlet characters."""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import pytest

from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_conductor,
)
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterConductorRequest,
)
from jacobian.math.number_theory.characters._tools import TOOLS
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character,
    dirichlet_character_table,
)


@pytest.mark.parametrize(
    ("modulus", "coordinates", "expected"),
    [
        (1, (), 1),
        (8, (0, 0), 1),
        (8, (0, 1), 8),
        (8, (1, 0), 4),
        (12, (0, 0), 1),
        (12, (1, 0), 4),
        (12, (0, 1), 3),
        (5, (1,), 5),
    ],
)
def test_known_conductors(modulus: int, coordinates: tuple[int, ...], expected: int):
    group = character_group(modulus)
    character = dirichlet_character(group, coordinates)

    result = dirichlet_character_conductor(character)

    assert result.character == character
    assert result.conductor == expected


def test_published_tool_executes_the_exact_conductor_contract() -> None:
    group = character_group(8)
    character = dirichlet_character(group, (0, 1))
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "dirichlet_character.conductor.compute"
    )

    result = tool.run(DirichletCharacterConductorRequest(character=character))

    assert result.character == character
    assert result.conductor == 8


def test_primitive_ancestor_survives_serialization_and_remains_composable() -> None:
    source = dirichlet_character(character_group(8), (0, 1))
    result = dirichlet_character_conductor(source)
    restored = type(result).model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.primitive_character.conductor == result.conductor
    assert dirichlet_character_table(restored.primitive_character.character) == (
        dirichlet_character_table(result.primitive_character.character)
    )


def _factors_through_by_value_table(character, divisor: int) -> bool:
    """Independent oracle: character values agree on every reduction fiber."""
    values = {}
    for residue, row in zip(
        character.group.unit_residues,
        character.group.unit_coordinates,
        strict=True,
    ):
        exponent = (
            sum(
                coordinate * (character.group.exponent // order) * coordinate_value
                for coordinate, order, coordinate_value in zip(
                    character.coordinates,
                    character.group.generator_orders,
                    row,
                    strict=True,
                )
            )
            % character.group.exponent
        )
        value = exponent
        reduced_residue = residue % divisor
        if reduced_residue in values and values[reduced_residue] != value:
            return False
        values[reduced_residue] = value
    return True


@pytest.mark.parametrize("modulus", range(1, 25))
def test_conductor_matches_independent_minimal_reduction_fiber_oracle(modulus: int):
    group = character_group(modulus)
    for serial in range(group.character_count):
        remaining = serial
        coordinates = []
        for order in group.generator_orders:
            coordinates.append(remaining % order)
            remaining //= order
        character = dirichlet_character(group, tuple(coordinates))
        actual = dirichlet_character_conductor(character).conductor
        result = dirichlet_character_conductor(character)

        fitting_divisors = [
            divisor
            for divisor in range(1, modulus + 1)
            if modulus % divisor == 0
            and _factors_through_by_value_table(character, divisor)
        ]
        assert actual == min(fitting_divisors)
        primitive = result.primitive_character
        assert primitive.conductor == actual
        assert dirichlet_character_conductor(primitive.character).conductor == actual
        primitive = primitive.character
        source_table = dirichlet_character_table(character)
        primitive_table = dirichlet_character_table(primitive)
        primitive_values = dict(
            zip(primitive_table.residues, primitive_table.values, strict=True)
        )
        for residue, source_value in zip(
            source_table.residues, source_table.values, strict=True
        ):
            if gcd(residue, modulus) != 1:
                continue
            target_value = primitive_values[residue % actual]
            assert source_value is not None and target_value is not None
            assert Fraction(source_value.exponent, source_value.order) == Fraction(
                target_value.exponent, target_value.order
            )
