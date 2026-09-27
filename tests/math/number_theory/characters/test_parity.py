from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_group_enumerate,
    dirichlet_character_parity,
)
from jacobian.math.number_theory.characters.operations import dirichlet_character_value
from jacobian.math.number_theory.characters.values import DirichletCharacter


def test_character_parity_is_exactly_its_value_at_minus_one():
    for modulus in (1, 3, 5, 8, 12, 15):
        group = character_group(modulus)
        family = dirichlet_character_group_enumerate(group)
        for coordinates in family.coordinates:
            character = DirichletCharacter(group=group, coordinates=coordinates)
            result = dirichlet_character_parity(character)
            value = dirichlet_character_value(character, -1).value

            assert value is not None
            assert result.character == character
            assert result.value == value
            assert (result.parity == "EVEN") == (value.exponent == 0)
