"""Exact length-generating polynomials for finite Weyl groups."""

import pytest

from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    CartanMatrixRequest,
    WeylPoincarePolynomialResult,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import (
    positive_roots,
    weyl_group_order,
    weyl_poincare_polynomial,
)


def _enumerated_length_profile(cartan):
    """Independent BFS on root-lattice action matrices for tiny groups."""
    rank = len(cartan)
    identity = tuple(
        tuple(int(row == column) for row in range(rank)) for column in range(rank)
    )
    lengths = {identity: 0}
    frontier = [identity]
    for element in frontier:
        for simple_index in range(rank):
            image_columns = []
            for column in element:
                pairing = sum(
                    column[index] * cartan[simple_index][index] for index in range(rank)
                )
                image = list(column)
                image[simple_index] -= pairing
                image_columns.append(tuple(image))
            image_element = tuple(image_columns)
            if image_element not in lengths:
                lengths[image_element] = lengths[element] + 1
                frontier.append(image_element)
    coefficients = [0] * (max(lengths.values()) + 1)
    for length in lengths.values():
        coefficients[length] += 1
    return tuple(coefficients)


@pytest.mark.parametrize(
    ("matrix", "coefficients"),
    (
        (((2,),), (1, 1)),
        (((2, -1), (-1, 2)), (1, 2, 2, 1)),
        (((2, -2), (-1, 2)), (1, 2, 2, 2, 1)),
        (((2, -3), (-1, 2)), (1, 2, 2, 2, 2, 2, 1)),
    ),
)
def test_small_weyl_poincare_polynomials_match_exact_length_profiles(
    matrix, coefficients
):
    cartan = CartanMatrix.model_validate(matrix)
    result = weyl_poincare_polynomial(cartan)
    assert result.polynomial.coefficients == coefficients
    assert tuple(
        reversed(result.polynomial.coefficients)
    ) == _enumerated_length_profile(matrix)
    assert sum(result.polynomial.coefficients) == weyl_group_order(cartan).group_order
    assert (
        WeylPoincarePolynomialResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_reducible_poincare_polynomial_is_the_product_of_factor_polynomials():
    a1_times_a2 = CartanMatrix.model_validate(((2, 0, 0), (0, 2, -1), (0, -1, 2)))
    result = weyl_poincare_polynomial(a1_times_a2)
    # (1 + q)(1 + 2q + 2q^2 + q^3)
    assert result.polynomial.coefficients == (1, 3, 4, 3, 1)
    assert sum(result.polynomial.coefficients) == 12


def test_e8_degree_equals_the_number_of_positive_roots():
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
    polynomial = weyl_poincare_polynomial(e8).polynomial.coefficients
    assert len(polynomial) == len(positive_roots(e8).positive_roots) + 1
    assert sum(polynomial) == weyl_group_order(e8).group_order
    assert polynomial[0] == polynomial[-1] == 1
    assert polynomial == tuple(reversed(polynomial))


def test_public_operation_and_example_use_the_parented_result():
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "root_system.weyl_poincare_polynomial.compute"
    )
    result = tool.run(
        CartanMatrixRequest(matrix=CartanMatrix.model_validate(((2, -1), (-1, 2))))
    )
    assert isinstance(result, WeylPoincarePolynomialResult)
    assert result.polynomial.coefficients == (1, 2, 2, 1)
