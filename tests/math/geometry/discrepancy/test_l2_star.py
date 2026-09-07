"""Independent cell integration establishes the discrepancy normalization."""

from fractions import Fraction
from itertools import pairwise, product
from math import prod

import pytest

from jacobian.math.geometry.discrepancy import squared_l2_star_discrepancy
from jacobian.math.matrices.values import rational_matrix_from_fractions


def _integral(points: list[tuple[Fraction, ...]], dimension: int) -> Fraction:
    axes = [
        sorted({Fraction(0), Fraction(1), *(p[j] for p in points)})
        for j in range(dimension)
    ]
    total = Fraction()
    n = len(points)
    for cell in product(*(tuple(pairwise(axis)) for axis in axes)):
        mid = [(a + b) / 2 for a, b in cell]
        count = sum(all(x < u for x, u in zip(p, mid, strict=True)) for p in points)
        total += count**2 * prod(b - a for a, b in cell)
        total -= 2 * n * count * prod((b * b - a * a) / 2 for a, b in cell)
        total += n * n * prod((b**3 - a**3) / 3 for a, b in cell)
    return total


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_exact_cell_integral(dimension: int) -> None:
    points = [
        tuple(Fraction(1, 4) for _ in range(dimension)),
        tuple(Fraction(3, 4) for _ in range(dimension)),
        tuple(Fraction(1, 4) for _ in range(dimension)),
        tuple(Fraction(j % 2) for j in range(dimension)),
    ]
    for count in range(5):
        prefix = points[:count]
        matrix = rational_matrix_from_fractions(tuple(prefix), column_count=dimension)
        assert squared_l2_star_discrepancy(matrix).as_fraction() == _integral(
            prefix, dimension
        )


def test_two_point_fixture() -> None:
    points = [(Fraction(1, 4), Fraction(1, 4)), (Fraction(3, 4), Fraction(3, 4))]
    assert squared_l2_star_discrepancy(
        rational_matrix_from_fractions(tuple(points))
    ).as_fraction() == Fraction(143, 1152)


@pytest.mark.parametrize("coordinate", [Fraction(-1, 10), Fraction(11, 10)])
def test_points_must_lie_in_unit_cube(coordinate: Fraction) -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError, match="lie in"):
        squared_l2_star_discrepancy(rational_matrix_from_fractions(((coordinate,),)))


def test_excessive_pair_work_is_rejected() -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    points = rational_matrix_from_fractions(((Fraction(0),),) * 1000)
    with pytest.raises(OperationDomainValidationError, match="1000000"):
        squared_l2_star_discrepancy(points)
