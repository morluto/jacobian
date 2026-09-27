from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character_group_enumerate,
    dirichlet_character_orthogonality,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
    DirichletCharacterFamily,
)


def test_character_family_is_complete_and_pairwise_orthogonal_for_small_moduli():
    for modulus in (1, 3, 5, 8, 12):
        group = character_group(modulus)
        family = dirichlet_character_group_enumerate(group)
        expected = tuple(product(*(range(order) for order in group.generator_orders)))

        assert family.group == group
        assert family.coordinates == expected
        assert len(family.coordinates) == group.character_count
        characters = tuple(
            DirichletCharacter(group=group, coordinates=coordinates)
            for coordinates in family.coordinates
        )
        for left_index, left in enumerate(characters):
            for right_index, right in enumerate(characters):
                pairing = dirichlet_character_orthogonality(left, right)
                assert pairing.value == (
                    group.character_count if left_index == right_index else 0
                )


def test_character_family_roundtrips_with_one_shared_group_parent():
    family = dirichlet_character_group_enumerate(character_group(20))

    decoded = DirichletCharacterFamily.model_validate_json(
        encode_strict_json(family.model_dump(mode="json"))
    )

    assert decoded == family
    assert (
        len(encode_strict_json(decoded.model_dump(mode="json")))
        < CanonicalLimits().max_output_bytes
    )


def test_character_family_rejects_duplicate_coordinate_rows():
    group = character_group(5)

    with pytest.raises(ValidationError, match="lexicographic order"):
        DirichletCharacterFamily(
            group=group,
            coordinates=((0,), (1,), (1,), (3,)),
        )


def test_character_family_accepts_the_maximum_modulus_with_bounded_output():
    group = character_group(2048)
    family = dirichlet_character_group_enumerate(group)
    encoded = encode_strict_json(family.model_dump(mode="json"))

    assert len(family.coordinates) == group.character_count
    assert family.coordinates[0] == (0,) * len(group.generator_orders)
    assert len(encoded) <= CanonicalLimits().max_output_bytes
