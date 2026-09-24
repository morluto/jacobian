"""Weyl exponents from positive-root height distributions."""

import pytest

from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    CartanMatrixRequest,
    WeylExponentsResult,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import weyl_exponents


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (((2,),), ((1,),)),
        (((2, -1), (-1, 2)), ((1, 2),)),
        (((2, -2), (-1, 2)), ((1, 3),)),
        (((2, -3), (-1, 2)), ((1, 5),)),
        (
            (
                (2, -1, 0),
                (-1, 2, -1),
                (0, -1, 2),
            ),
            ((1, 2, 3),),
        ),
    ],
)
def test_irreducible_exponents_match_independent_classical_values(matrix, expected):
    result = weyl_exponents(CartanMatrix.model_validate(matrix))
    assert tuple(component.exponents for component in result.components) == expected
    assert tuple(len(component.exponents) for component in result.components) == tuple(
        len(component.simple_root_indices) for component in result.components
    )
    assert WeylExponentsResult.model_validate_json(result.model_dump_json()) == result


def test_reducible_exponents_preserve_component_axes_and_multiplicity():
    a1_plus_a2 = CartanMatrix.model_validate(((2, 0, 0), (0, 2, -1), (0, -1, 2)))
    result = weyl_exponents(a1_plus_a2)
    assert tuple(component.simple_root_indices for component in result.components) == (
        (0,),
        (1, 2),
    )
    assert tuple(component.exponents for component in result.components) == (
        (1,),
        (1, 2),
    )


def test_e8_exponents_are_computed_from_the_full_root_height_profile():
    e8 = CartanMatrix.model_validate(
        (
            (2, -1, 0, 0, 0, 0, 0, 0),
            (-1, 2, -1, 0, 0, 0, 0, 0),
            (0, -1, 2, -1, 0, 0, 0, -1),
            (0, 0, -1, 2, -1, 0, 0, 0),
            (0, 0, 0, -1, 2, -1, 0, 0),
            (0, 0, 0, 0, -1, 2, -1, 0),
            (0, 0, 0, 0, 0, -1, 2, 0),
            (0, 0, -1, 0, 0, 0, 0, 2),
        )
    )
    assert weyl_exponents(e8).components[0].exponents == (1, 7, 11, 13, 17, 19, 23, 29)


def test_public_operation_returns_the_native_exponent_value():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "root_system.weyl_exponents.compute"
    )
    request = CartanMatrixRequest(
        matrix=CartanMatrix.model_validate(((2, -3), (-1, 2)))
    )
    result = tool.run(request)
    assert isinstance(result, WeylExponentsResult)
    assert result.components[0].exponents == (1, 5)
