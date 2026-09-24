import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_conjugate,
    dirichlet_character_group_enumerate,
    dirichlet_character_power,
)
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_product,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter


def test_signed_character_power_matches_repeated_group_multiplication():
    for modulus in (1, 3, 5, 8, 12, 15):
        group = character_group(modulus)
        for coordinates in dirichlet_character_group_enumerate(group).coordinates:
            character = DirichletCharacter(group=group, coordinates=coordinates)
            for exponent in range(-3, 4):
                factor = (
                    character
                    if exponent >= 0
                    else dirichlet_character_conjugate(character)
                )
                result = DirichletCharacter(
                    group=group, coordinates=(0,) * len(coordinates)
                )
                for _ in range(abs(exponent)):
                    result = dirichlet_character_product(result, factor)
                assert dirichlet_character_power(character, exponent) == result


def test_character_power_rejects_excessive_exponent_digits_before_scaling():
    character = DirichletCharacter(group=character_group(5), coordinates=(1,))

    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_power(character, 10**256)

    assert error.value.errors()[0]["type"] == "dirichlet_character.integer_digit_bound"
