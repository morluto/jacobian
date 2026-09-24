from __future__ import annotations

from math import lcm

import pytest
from pydantic_core import PydanticCustomError
from sympy import I, Rational, exp, pi, to_number_field

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.cyclic_linear._models import RationalCyclotomicField
from jacobian.math.number_theory.characters._models import (
    DirichletCharacterGeneralizedGaussSumRequest,
    DirichletCharacterGeneralizedGaussSumResult,
)
from jacobian.math.number_theory.characters._tools import TOOLS
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_generalized_gauss_sum,
)


def _character_values(coordinates: tuple[int, ...]) -> dict[int, int]:
    if coordinates == (0,):
        return {1: 1, 2: 1, 3: 1, 4: 1}
    assert coordinates == (2,)
    return {1: 1, 2: -1, 3: -1, 4: 1}


@pytest.mark.parametrize("frequency", (0, 1, 2, 5, -3))
@pytest.mark.parametrize("coordinates", ((0,), (2,)))
def test_generalized_gauss_sum_matches_direct_exact_residue_sum(
    frequency: int, coordinates: tuple[int, ...]
) -> None:
    character = dirichlet_character(character_group(5), coordinates)
    result = dirichlet_character_generalized_gauss_sum(character, frequency)
    values = _character_values(coordinates)
    value_order = 1 if coordinates == (0,) else 2
    field_order = lcm(5, value_order)
    direct = sum(
        value * exp(2 * pi * I * frequency * residue / 5)
        for residue, value in values.items()
    )
    zeta = exp(2 * pi * I / field_order)
    represented = sum(
        Rational(coefficient.num, coefficient.den) * zeta**power
        for power, coefficient in enumerate(result.value.coefficients_ascending)
    )

    assert result.character == character
    assert result.frequency == frequency
    assert result.frequency_residue == frequency % 5
    assert result.value.field == RationalCyclotomicField(order=field_order)
    assert to_number_field(represented - direct, zeta).as_expr() == 0
    assert (
        DirichletCharacterGeneralizedGaussSumResult.model_validate_json(
            result.model_dump_json()
        )
        == result
    )


def test_generalized_gauss_sum_is_discoverable_and_composes_through_catalog() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "dirichlet_character.generalized_gauss_sum.compute"
    )
    request = tool.request_type(
        character=dirichlet_character(character_group(5), (2,)), frequency=0
    )
    assert isinstance(request, DirichletCharacterGeneralizedGaussSumRequest)
    result = tool.run(request)
    assert all(
        coefficient.as_fraction() == 0
        for coefficient in result.value.coefficients_ascending
    )
    assert (
        Catalog(TOOLS).operation("dirichlet_character.generalized_gauss_sum.compute")
        is tool
    )


def test_generalized_gauss_sum_preserves_principal_nonunit_frequency() -> None:
    principal = dirichlet_character(character_group(5), (0,))
    result = dirichlet_character_generalized_gauss_sum(principal, 0)
    assert tuple(
        value.as_fraction() for value in result.value.coefficients_ascending
    ) == (
        Rational(4),
        Rational(0),
        Rational(0),
        Rational(0),
    )


def test_generalized_gauss_sum_admits_frequency_and_field_before_sum() -> None:
    character = dirichlet_character(character_group(3), (0,))
    with pytest.raises(PydanticCustomError):
        dirichlet_character_generalized_gauss_sum(character, 10**256)

    large_character = dirichlet_character(character_group(257), (0,))
    with pytest.raises(OperationResourceAdmissionError):
        dirichlet_character_generalized_gauss_sum(large_character, 0)
