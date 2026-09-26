from __future__ import annotations

import json
from fractions import Fraction

import pytest

from jacobian.math.number_theory.characters import _tools
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterLValueNonpositiveRequest,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_l_value_nonpositive_integer,
)


def _coefficients(result) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(value.num, value.den) for value in result.value.coefficients_ascending
    )


def test_quadratic_character_modulo_three_has_exact_l_zero() -> None:
    # Direct finite formula: B_1(x)=x-1/2 and chi(1)=1, chi(2)=-1.
    b1 = Fraction(1, 3) - Fraction(1, 2) - (Fraction(2, 3) - Fraction(1, 2))
    assert b1 == Fraction(-1, 3)

    character = dirichlet_character(character_group(3), (1,))
    result = dirichlet_character_l_value_nonpositive_integer(character, 1)
    assert result.character == character
    assert result.bernoulli_index == 1
    assert result.argument == 0
    assert result.value.field.order == 2
    assert _coefficients(result) == (Fraction(1, 3),)


def test_nonreal_character_value_is_exact_and_field_bound() -> None:
    # Evaluate the defining finite sum independently in Q(i).
    values = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
        (Fraction(-1), Fraction(0)),
    )
    bernoulli_components = [Fraction(0), Fraction(0)]
    for integer, (real, imaginary) in enumerate(values, start=1):
        weight = Fraction(integer, 5) - Fraction(1, 2)
        bernoulli_components[0] += real * weight
        bernoulli_components[1] += imaginary * weight
    assert tuple(bernoulli_components) == (Fraction(-3, 5), Fraction(-1, 5))

    # Thus L(0,chi)=-B(1,chi)=(3+i)/5 in Q(i).
    character = dirichlet_character(character_group(5), (1,))
    result = dirichlet_character_l_value_nonpositive_integer(character, 1)
    assert result.argument == 0
    assert result.value.field.order == 4
    assert _coefficients(result) == (Fraction(3, 5), Fraction(1, 5))


def test_trivial_modulus_endpoint_matches_zeta_zero() -> None:
    character = dirichlet_character(character_group(1), ())
    result = dirichlet_character_l_value_nonpositive_integer(character, 1)
    assert result.argument == 0
    assert _coefficients(result) == (Fraction(-1, 2),)


def test_request_and_catalog_example_round_trip() -> None:
    tool = next(
        tool
        for tool in _tools.TOOLS
        if tool.operation_id
        == "dirichlet_character.l_value_nonpositive_integer.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.argument == 0
    tool.result_type.model_validate_json(json.dumps(result.model_dump(mode="json")))


def test_only_nonpositive_integer_arguments_are_admitted() -> None:
    character = dirichlet_character(character_group(3), (1,))
    with pytest.raises(ValueError):
        DirichletCharacterLValueNonpositiveRequest(
            character=character,
            bernoulli_index=0,
        )
    assert (
        dirichlet_character_l_value_nonpositive_integer(character, 32).argument == -31
    )
