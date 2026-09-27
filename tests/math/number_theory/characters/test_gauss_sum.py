from __future__ import annotations

from math import lcm

import pytest
from sympy import I, Rational, exp, pi, to_number_field

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterGaussSumResult,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_gauss_sum,
)


def _direct_root_sum(modulus: int, coordinates: tuple[int, ...]):
    if modulus == 3 and coordinates == (0,):
        value_order, values = 1, {1: 1, 2: 1}
    elif modulus == 5 and coordinates == (2,):
        value_order, values = 2, {1: 1, 2: -1, 3: -1, 4: 1}
    else:
        # For 2 as a generator modulo 5, the coordinate-one character takes
        # values 1, i, -i, -1 on 1, 2, 4, 3 respectively.
        value_order, values = 4, {1: 1, 2: I, 3: -I, 4: -1}
    field_order = lcm(modulus, value_order)
    direct_sum = sum(
        value * exp(2 * pi * I * residue / modulus) for residue, value in values.items()
    )
    return field_order, direct_sum


@pytest.mark.parametrize(
    ("modulus", "coordinates"),
    ((3, (0,)), (5, (2,)), (5, (1,))),
    ids=("principal", "quadratic", "nonreal-quartic"),
)
def test_gauss_sum_matches_independent_exact_root_of_unity_sum(modulus, coordinates):
    character = dirichlet_character(character_group(modulus), coordinates)
    result = dirichlet_character_gauss_sum(character)
    field_order, direct_sum = _direct_root_sum(modulus, coordinates)
    assert result.character == character
    assert result.value.field.order == field_order
    assert (
        DirichletCharacterGaussSumResult.model_validate(
            result.model_dump(mode="python")
        )
        == result
    )

    zeta = exp(2 * pi * I / field_order)
    represented = sum(
        Rational(coefficient.num, coefficient.den) * zeta**power
        for power, coefficient in enumerate(result.value.coefficients_ascending)
    )
    assert to_number_field(represented - direct_sum, zeta).as_expr() == 0


def test_gauss_sum_admits_target_field_order_before_construction():
    character = dirichlet_character(character_group(257), (0,))
    with pytest.raises(OperationResourceAdmissionError) as error:
        dirichlet_character_gauss_sum(character)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.gauss_sum.cyclotomic_order_bound"
    )
