"""Exact contract tests for the published character-conjugation operation."""

from itertools import product

from jacobian.math.number_theory.characters._models import (
    DirichletCharacterConjugateRequest,
)
from jacobian.math.number_theory.characters._tools import (
    TOOLS,
    _compute_character_conjugate,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_product,
    dirichlet_character_table,
)
from jacobian.math.number_theory.characters.values import DirichletCharacter


def test_conjugation_negates_exact_character_values_and_is_involutive() -> None:
    group = character_group(5)
    character = dirichlet_character(group, (1,))
    conjugate = _compute_character_conjugate(
        DirichletCharacterConjugateRequest(character=character)
    )

    original_table = dirichlet_character_table(character)
    conjugate_table = dirichlet_character_table(conjugate)
    for original, actual in zip(
        original_table.values, conjugate_table.values, strict=True
    ):
        assert actual == (None if original is None else original.conjugate())

    assert conjugate.group == character.group
    serialized = conjugate.model_dump_json()
    assert DirichletCharacter.model_validate_json(serialized) == conjugate
    assert (
        dirichlet_character_table(DirichletCharacter.model_validate_json(serialized))
        == conjugate_table
    )
    assert (
        _compute_character_conjugate(
            DirichletCharacterConjugateRequest(character=conjugate)
        )
        == character
    )
    identity = dirichlet_character_product(character, conjugate)
    assert identity.coordinates == (0,)


def test_conjugation_is_exact_for_every_character_in_small_groups() -> None:
    for modulus in (1, 2, 3, 4, 5, 8, 12):
        group = character_group(modulus)
        coordinate_domain = tuple(range(order) for order in group.generator_orders)
        all_coordinates = product(*coordinate_domain) if coordinate_domain else [()]
        for coordinates in all_coordinates:
            character = dirichlet_character(group, tuple(coordinates))
            conjugate = _compute_character_conjugate(
                DirichletCharacterConjugateRequest(character=character)
            )
            lhs = dirichlet_character_table(conjugate).values
            rhs = tuple(
                None if value is None else value.conjugate()
                for value in dirichlet_character_table(character).values
            )
            assert lhs == rhs


def test_conjugation_operation_is_published_with_canonical_result_type() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "dirichlet_character.conjugate.compute"
    )
    assert tool.request_type is DirichletCharacterConjugateRequest
    assert tool.result_type.__name__ == "DirichletCharacter"
    example = tool.examples[0]
    request = tool.request_type.model_validate(example.input)
    result = tool.run(request)
    assert result == dirichlet_character(character_group(5), (3,))
