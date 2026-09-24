from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterSequenceTwistRequest,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_product,
    dirichlet_character_sequence_twist,
)
from jacobian.math.number_theory.sequences.core import (
    FiniteCyclotomicSequence,
    FiniteIntegerSequence,
)


def _coefficients(value: RationalCyclotomicElement) -> tuple[Fraction, ...]:
    return tuple(Fraction(item.num, item.den) for item in value.coefficients_ascending)


def test_quartic_character_twist_matches_independent_gaussian_value_table() -> None:
    group = character_group(5)
    character = dirichlet_character(group, (1,))
    result = dirichlet_character_sequence_twist(
        DirichletCharacterSequenceTwistRequest(
            sequence=FiniteIntegerSequence(values=(1, 1, 1, 1)),
            character=character,
            index_origin=1,
        )
    )

    assert result.index_origin == 1
    assert result.field.order == 4
    assert tuple(_coefficients(value) for value in result.values) == (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
        (Fraction(-1), Fraction(0)),
    )


def test_twists_compose_and_preserve_the_authored_axis_through_json() -> None:
    group = character_group(5)
    character = dirichlet_character(group, (1,))
    first = dirichlet_character_sequence_twist(
        DirichletCharacterSequenceTwistRequest(
            sequence=FiniteIntegerSequence(values=(1, 1, 1, 1)),
            character=character,
            index_origin=1,
        )
    )
    roundtripped = FiniteCyclotomicSequence.model_validate_json(first.model_dump_json())
    assert roundtripped == first

    twice = dirichlet_character_sequence_twist(
        DirichletCharacterSequenceTwistRequest(
            sequence=roundtripped,
            character=character,
        )
    )
    squared_character = dirichlet_character_product(character, character)
    direct = dirichlet_character_sequence_twist(
        DirichletCharacterSequenceTwistRequest(
            sequence=FiniteIntegerSequence(values=(1, 1, 1, 1)),
            character=squared_character,
            index_origin=1,
        )
    )

    # The repeated twist is represented in Q(zeta_4), while chi^2 is represented
    # in its smaller Q(zeta_2) field. Their exact values agree as integers.
    assert twice.index_origin == first.index_origin == direct.index_origin == 1
    assert tuple(int(_coefficients(value)[0]) for value in twice.values) == (
        1,
        -1,
        -1,
        1,
    )
    assert tuple(int(_coefficients(value)[0]) for value in direct.values) == (
        1,
        -1,
        -1,
        1,
    )


def test_twisting_existing_cyclotomic_sequence_uses_exact_common_field_embedding() -> (
    None
):
    source_field = RationalCyclotomicField(order=3)
    source = FiniteCyclotomicSequence(
        index_origin=1,
        field=source_field,
        values=(
            RationalCyclotomicElement(
                field=source_field,
                coefficients_ascending=(
                    CanonicalRational(num=0, den=1),
                    CanonicalRational(num=1, den=1),
                ),
            ),
        ),
    )
    quadratic_mod5 = dirichlet_character(character_group(5), (2,))
    result = dirichlet_character_sequence_twist(
        DirichletCharacterSequenceTwistRequest(
            sequence=source, character=quadratic_mod5
        )
    )

    # Q(zeta_3) embeds in Q(zeta_6) by zeta_3 -> zeta_6^2; the index 1
    # character value is one, so this checks the source-field embedding itself.
    assert result.field.order == 6
    assert result.index_origin == 1
    assert _coefficients(result.values[0]) == (Fraction(-1), Fraction(1))


def test_twist_preflights_field_order_and_serialized_output_size() -> None:
    too_large_field_character = dirichlet_character(character_group(509), (1,))
    with pytest.raises(OperationResourceAdmissionError) as field_error:
        dirichlet_character_sequence_twist(
            DirichletCharacterSequenceTwistRequest(
                sequence=FiniteIntegerSequence(values=(1,)),
                character=too_large_field_character,
                index_origin=1,
            )
        )
    assert (
        field_error.value.errors()[0]["type"]
        == "dirichlet_character.sequence_twist.field_order_bound"
    )

    quartic_character = dirichlet_character(character_group(5), (1,))
    oversized_sequence = FiniteIntegerSequence(values=(1,) * 30_000)
    with pytest.raises(OperationResourceAdmissionError) as size_error:
        dirichlet_character_sequence_twist(
            DirichletCharacterSequenceTwistRequest(
                sequence=oversized_sequence,
                character=quartic_character,
                index_origin=1,
            )
        )
    assert (
        size_error.value.errors()[0]["type"]
        == "dirichlet_character.sequence_twist.result_bytes_bound"
    )
