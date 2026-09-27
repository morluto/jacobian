"""Exact contract tests for Dirichlet-character inversion."""

from jacobian.math.number_theory.characters import (
    character_group,
    dirichlet_character_conjugate,
    dirichlet_character_group_enumerate,
    dirichlet_character_inverse,
)
from jacobian.math.number_theory.characters.operations import (
    dirichlet_character_product,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter


def test_inverse_is_the_dual_group_inverse_and_character_conjugate() -> None:
    for modulus in (1, 2, 3, 4, 5, 8, 12, 15):
        group = character_group(modulus)
        family = dirichlet_character_group_enumerate(group)
        identity = DirichletCharacter(
            group=group, coordinates=(0,) * len(group.generator_orders)
        )
        for coordinates in family.coordinates:
            character = DirichletCharacter(group=group, coordinates=coordinates)
            inverse = dirichlet_character_inverse(character)

            assert inverse.group == character.group
            assert inverse == dirichlet_character_conjugate(character)
            assert dirichlet_character_product(character, inverse) == identity
            assert (
                DirichletCharacter.model_validate_json(inverse.model_dump_json())
                == inverse
            )
