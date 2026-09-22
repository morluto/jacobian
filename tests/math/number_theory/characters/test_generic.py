import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_product,
    dirichlet_character_table,
    dirichlet_character_value,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
    DirichletCharacterGroup,
)


def test_generic_character_is_multiplicative_and_zero_off_units() -> None:
    group = character_group(5)
    character = dirichlet_character(group, (1,))
    value_two = dirichlet_character_value(character, 2).value
    value_four = dirichlet_character_value(character, 4).value
    assert value_two is not None and value_four is not None
    assert value_two.multiply(value_two) == value_four
    assert dirichlet_character_value(character, 5).value is None
    square = dirichlet_character_product(character, character)
    square_value = dirichlet_character_value(square, 2).value
    assert square_value is not None and square_value.exponent == 2
    assert len(dirichlet_character_table(character).values) == 5


def test_native_character_value_rejects_the_smallest_overbound_integer() -> None:
    character = dirichlet_character(character_group(3), (0,))

    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_value(character, int("1" * 257))
    assert error.value.errors()[0]["type"] == "dirichlet_character.integer_digit_bound"


def test_native_character_consumer_rejects_missing_group_fields() -> None:
    missing_group = DirichletCharacter.model_construct(coordinates=())
    with pytest.raises(OperationDomainValidationError):
        dirichlet_character_value(missing_group, 1)

    valid = character_group(3)
    malformed_group = DirichletCharacterGroup.model_construct(
        **{**valid.model_dump(), "generator_orders": None}
    )
    malformed = DirichletCharacter.model_construct(
        group=malformed_group, coordinates=()
    )
    with pytest.raises(OperationDomainValidationError):
        dirichlet_character_table(malformed)


def test_character_parent_mismatch_rejected() -> None:
    first = dirichlet_character(character_group(5), (1,))
    second = dirichlet_character(character_group(7), (1,))
    with pytest.raises(OperationDomainValidationError):
        dirichlet_character_product(first, second)


def test_product_rechecks_a_forged_serialized_group_parent() -> None:
    valid = character_group(3)
    forged_group = DirichletCharacterGroup.model_construct(
        modulus=3,
        unit_residues=(1,),
        character_count=1,
        invariant_factors=(),
        generators=(),
        generator_orders=(),
        unit_coordinates=(((),),),
        exponent=1,
    )
    forged = DirichletCharacter.model_construct(group=forged_group, coordinates=())
    with pytest.raises(OperationDomainValidationError) as error:
        dirichlet_character_product(forged, forged)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.group.unit_residues_mismatch"
    )
    # Keep a valid parent as an independent positive control.
    assert (
        dirichlet_character_product(
            dirichlet_character(valid, (0,)), dirichlet_character(valid, (0,))
        ).group
        == valid
    )
