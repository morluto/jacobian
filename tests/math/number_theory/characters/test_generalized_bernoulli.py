from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.characters import _tools
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterGeneralizedBernoulliPrefixRequest,
    DirichletCharacterGeneralizedBernoulliRequest,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_generalized_bernoulli,
    dirichlet_character_generalized_bernoulli_prefix,
)


def _bernoulli_numbers(index: int) -> tuple[Fraction, ...]:
    """Independent rational recurrence, with classical B_1=-1/2."""

    values = [Fraction(1)]
    for n in range(1, index + 1):
        values.append(-sum(math.comb(n + 1, j) * values[j] for j in range(n)) / (n + 1))
    return tuple(values)


def _direct_formula(
    *,
    modulus: int,
    index: int,
    character_values: dict[int, tuple[Fraction, Fraction]],
) -> tuple[Fraction, Fraction]:
    """Evaluate N^(k-1) sum chi(a) B_k(a/N) directly in Q(i)."""

    bernoulli = _bernoulli_numbers(index)
    total = [Fraction(0), Fraction(0)]
    for integer in range(1, modulus + 1):
        char_value = character_values.get(integer % modulus, (Fraction(0), Fraction(0)))
        if char_value == (0, 0):
            continue
        x = Fraction(integer, modulus)
        polynomial_value = sum(
            math.comb(index, j) * bernoulli[j] * x ** (index - j)
            for j in range(index + 1)
        )
        scale = Fraction(1, modulus) if index == 0 else Fraction(modulus ** (index - 1))
        weight = scale * polynomial_value
        total[0] += char_value[0] * weight
        total[1] += char_value[1] * weight
    return total[0], total[1]


def _result_coefficients(result) -> tuple[Fraction, ...]:
    return tuple(Fraction(c.num, c.den) for c in result.value.coefficients_ascending)


def test_principal_character_low_indices_match_direct_rational_formula() -> None:
    character = dirichlet_character(character_group(5), (0,))
    values = {
        1: (Fraction(1), Fraction(0)),
        2: (Fraction(1), Fraction(0)),
        3: (Fraction(1), Fraction(0)),
        4: (Fraction(1), Fraction(0)),
    }

    for index, expected in (
        (0, Fraction(4, 5)),
        (1, Fraction(0)),
        (2, Fraction(-2, 3)),
    ):
        result = dirichlet_character_generalized_bernoulli(character, index)
        assert _direct_formula(modulus=5, index=index, character_values=values) == (
            expected,
            Fraction(0),
        )
        assert result.character == character
        assert result.index == index
        assert result.value.field.order == 1
        assert _result_coefficients(result) == (expected,)


def test_prefix_matches_independent_direct_formula_and_scalar_operations() -> None:
    character = dirichlet_character(character_group(5), (0,))
    prefix = dirichlet_character_generalized_bernoulli_prefix(character, 2)
    assert prefix.character == character
    assert prefix.maximum_index == 2
    expected = (Fraction(4, 5), Fraction(0), Fraction(-2, 3))
    assert tuple(
        tuple(Fraction(c.num, c.den) for c in value.coefficients_ascending)
        for value in prefix.values
    ) == tuple((value,) for value in expected)
    assert prefix.values == tuple(
        dirichlet_character_generalized_bernoulli(character, index).value
        for index in range(3)
    )


def test_prefix_request_and_catalog_example_round_trip() -> None:
    request = DirichletCharacterGeneralizedBernoulliPrefixRequest(
        character=dirichlet_character(character_group(3), (0,)), maximum_index=2
    )
    assert request.maximum_index == 2
    tool = next(
        tool
        for tool in _tools.TOOLS
        if tool.operation_id
        == "dirichlet_character.generalized_bernoulli_prefix.compute"
    )
    parsed = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(parsed)
    assert len(result.values) == 3
    tool.result_type.model_validate_json(json.dumps(result.model_dump(mode="json")))


def test_trivial_modulus_one_uses_bernoulli_polynomial_endpoint_convention() -> None:
    character = dirichlet_character(character_group(1), ())
    values = {0: (Fraction(1), Fraction(0))}
    result_zero = dirichlet_character_generalized_bernoulli(character, 0)
    result_one = dirichlet_character_generalized_bernoulli(character, 1)

    assert _direct_formula(modulus=1, index=0, character_values=values) == (
        Fraction(1),
        Fraction(0),
    )
    assert _direct_formula(modulus=1, index=1, character_values=values) == (
        Fraction(1, 2),
        Fraction(0),
    )
    assert _result_coefficients(result_zero) == (Fraction(1),)
    assert _result_coefficients(result_one) == (Fraction(1, 2),)


def test_parity_forces_an_exact_zero_against_direct_formula() -> None:
    character = dirichlet_character(character_group(3), (1,))
    values = {1: (Fraction(1), Fraction(0)), 2: (Fraction(-1), Fraction(0))}
    result = dirichlet_character_generalized_bernoulli(character, 2)
    assert _direct_formula(modulus=3, index=2, character_values=values) == (
        Fraction(0),
        Fraction(0),
    )
    assert _result_coefficients(result) == (Fraction(0),)


def test_nonreal_character_matches_direct_gaussian_rational_oracle() -> None:
    character = dirichlet_character(character_group(5), (1,))
    values = {
        1: (Fraction(1), Fraction(0)),
        2: (Fraction(0), Fraction(1)),
        3: (Fraction(0), Fraction(-1)),
        4: (Fraction(-1), Fraction(0)),
    }
    expected = _direct_formula(modulus=5, index=1, character_values=values)
    result = dirichlet_character_generalized_bernoulli(character, 1)

    assert expected == (Fraction(-3, 5), Fraction(-1, 5))
    assert result.value.field.order == 4
    assert _result_coefficients(result) == expected


def test_nonreal_prefix_preserves_the_character_value_field() -> None:
    character = dirichlet_character(character_group(5), (1,))
    character_values = {
        1: (Fraction(1), Fraction(0)),
        2: (Fraction(0), Fraction(1)),
        3: (Fraction(0), Fraction(-1)),
        4: (Fraction(-1), Fraction(0)),
    }
    prefix = dirichlet_character_generalized_bernoulli_prefix(character, 1)
    assert prefix.values[0].field.order == 4
    assert tuple(
        tuple(Fraction(c.num, c.den) for c in value.coefficients_ascending)
        for value in prefix.values
    ) == tuple(
        _direct_formula(modulus=5, index=index, character_values=character_values)
        for index in range(2)
    )


def test_index_bound_and_field_order_are_admitted_before_expansion() -> None:
    character = dirichlet_character(character_group(1), ())
    boundary_request = DirichletCharacterGeneralizedBernoulliRequest(
        character=character,
        index=32,
    )
    assert (
        dirichlet_character_generalized_bernoulli(
            boundary_request.character, boundary_request.index
        ).index
        == 32
    )
    with pytest.raises(ValueError):
        DirichletCharacterGeneralizedBernoulliRequest(
            character=character,
            index=33,
        )

    high_order_character = dirichlet_character(character_group(509), (1,))
    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_generalized_bernoulli(high_order_character, 1)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.generalized_bernoulli.field_order_bound"
    )


def test_coefficient_growth_is_rejected_before_bernoulli_expansion() -> None:
    # Modulo 2*1009 the unit group is cyclic of order 1008. Coordinate 8
    # defines a character of exact order 126; at index 32 the conservative
    # reduced-coordinate bound exceeds the 256-digit coefficient envelope.
    character = dirichlet_character(character_group(2018), (8,))
    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_generalized_bernoulli(character, 32)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.generalized_bernoulli.coefficient_bound"
    )


def test_catalog_example_executes_and_round_trips() -> None:
    tool = next(
        tool
        for tool in _tools.TOOLS
        if tool.operation_id == "dirichlet_character.generalized_bernoulli.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert _result_coefficients(result) == (Fraction(2, 3),)
    tool.result_type.model_validate_json(json.dumps(result.model_dump(mode="json")))
