import pytest
from sympy import I, Rational, exp, pi, to_number_field

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters._tools import TOOLS
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_primitive_gauss_norm,
)


@pytest.mark.parametrize(
    ("modulus", "coordinates", "values"),
    (
        (3, (1,), {1: 1, 2: -1}),
        (5, (2,), {1: 1, 2: -1, 3: -1, 4: 1}),
        (5, (1,), {1: 1, 2: I, 3: -I, 4: -1}),
    ),
)
def test_primitive_gauss_norm_matches_independent_complex_norm(
    modulus, coordinates, values
):
    character = dirichlet_character(character_group(modulus), coordinates)
    result = dirichlet_character_primitive_gauss_norm(character)
    tau = sum(
        value * exp(2 * pi * I * residue / modulus) for residue, value in values.items()
    )
    zeta = exp(2 * pi * I / result.gauss_sum.field.order)
    represented = sum(
        Rational(coefficient.num, coefficient.den) * zeta**power
        for power, coefficient in enumerate(result.gauss_sum.coefficients_ascending)
    )
    assert to_number_field(represented - tau, zeta).as_expr() == 0
    assert result.conductor == modulus
    assert result.norm_squared.coefficients_ascending[0].num == modulus
    assert all(
        value.num == 0 for value in result.norm_squared.coefficients_ascending[1:]
    )
    assert to_number_field(tau * tau.conjugate() - modulus, zeta).as_expr() == 0


def test_primitive_gauss_norm_rejects_imprimitive_character():
    principal = dirichlet_character(character_group(5), (0,))
    with pytest.raises(OperationDomainValidationError) as error:
        dirichlet_character_primitive_gauss_norm(principal)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.primitive_gauss_norm.requires_primitive"
    )


def test_primitive_gauss_norm_is_discoverable_and_runs_from_its_typed_request():
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "dirichlet_character.primitive_gauss_norm.compute"
    )
    request = tool.request_type(character=dirichlet_character(character_group(5), (2,)))
    result = tool.run(request)
    assert result.norm_squared.coefficients_ascending[0].num == 5
