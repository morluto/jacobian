from itertools import permutations
from math import prod

import pytest

from jacobian.math.groups.root_systems import coxeter_polynomial


def _determinant(matrix: tuple[tuple[int, ...], ...]) -> int:
    size = len(matrix)
    return sum(
        (-1) ** sum(a > b for i, a in enumerate(order) for b in order[i + 1 :])
        * prod(matrix[row][column] for row, column in enumerate(order))
        for order in permutations(range(size))
    )


def _direct_coxeter_matrix(
    cartan: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    rank = len(cartan)
    product_matrix = [
        [int(row == column) for column in range(rank)] for row in range(rank)
    ]
    for index in range(rank):
        reflection = [
            [int(row == column) for column in range(rank)] for row in range(rank)
        ]
        reflection[index] = [
            int(index == column) - cartan[index][column] for column in range(rank)
        ]
        product_matrix = [
            [
                sum(reflection[i][k] * product_matrix[k][j] for k in range(rank))
                for j in range(rank)
            ]
            for i in range(rank)
        ]
    return tuple(tuple(row) for row in product_matrix)


@pytest.mark.parametrize(
    ("cartan", "coefficients"),
    [
        (((2,),), (1, 1)),  # A1: t+1
        (((2, -1), (-1, 2)), (1, 1, 1)),  # A2: t^2+t+1
        (((2, -2), (-1, 2)), (1, 0, 1)),  # B2: t^2+1
        (((2, 0, 0), (0, 2, -1), (0, -1, 2)), (1, 2, 2, 1)),  # A1 x A2
    ],
)
def test_known_coxeter_polynomials_and_direct_determinant(cartan, coefficients):
    polynomial = coxeter_polynomial(cartan)
    assert polynomial.coefficients == coefficients

    coxeter = _direct_coxeter_matrix(cartan)
    for value in range(-2, 4):
        evaluated = sum(
            coefficient * value ** (len(coefficients) - index - 1)
            for index, coefficient in enumerate(coefficients)
        )
        characteristic_matrix = tuple(
            tuple(
                int(row == column) * value - coxeter[row][column]
                for column in range(len(cartan))
            )
            for row in range(len(cartan))
        )
        assert evaluated == _determinant(characteristic_matrix)
