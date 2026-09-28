from math import gcd, lcm

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_group_enumerate,
    dirichlet_character_order,
)
from jacobian.math.number_theory.characters._models import DirichletCharacterOrderResult
from jacobian.math.number_theory.characters.operations import dirichlet_character_value
from jacobian.math.number_theory.characters.values import DirichletCharacter


def test_character_order_matches_exact_orders_of_all_unit_values():
    for modulus in (1, 3, 5, 8, 12, 15):
        group = character_group(modulus)
        family = dirichlet_character_group_enumerate(group)
        for coordinates in family.coordinates:
            character = DirichletCharacter(group=group, coordinates=coordinates)
            coordinate_order = lcm(
                *(
                    axis_order // gcd(coordinate, axis_order)
                    for coordinate, axis_order in zip(
                        coordinates, group.generator_orders, strict=True
                    )
                )
            )
            value_orders = []
            for residue in group.unit_residues:
                value = dirichlet_character_value(character, residue).value
                assert value is not None
                value_orders.append(
                    group.exponent // gcd(value.exponent, group.exponent)
                )

            assert dirichlet_character_order(character).order == lcm(*value_orders)
            assert coordinate_order == lcm(*value_orders)


def test_character_order_result_rejects_a_false_divisor_of_the_exponent():
    group = character_group(3)
    character = DirichletCharacter(group=group, coordinates=(1,))

    with pytest.raises(ValidationError, match="order induced by its dual coordinates"):
        DirichletCharacterOrderResult(character=character, order=1)
