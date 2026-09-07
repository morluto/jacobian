"""Direct sphere enumeration independently checks character contraction."""

from fractions import Fraction
from itertools import combinations
from typing import Literal

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.boolean.fourier import (
    RationalWalshPolynomial,
    WalshTerm,
    fixed_weight_moment,
)


def _polynomial(n: int) -> RationalWalshPolynomial:
    return RationalWalshPolynomial(
        variable_count=n,
        terms=(
            WalshTerm(character=(), coefficient=CanonicalRational(num=3, den=1)),
            WalshTerm(character=(0, 1), coefficient=CanonicalRational(num=2, den=1)),
            WalshTerm(
                character=(0, 1, 2, 3), coefficient=CanonicalRational(num=-5, den=1)
            ),
        ),
    )


@pytest.mark.parametrize("n", [4, 5, 8])
@pytest.mark.parametrize("order", [1, 2])
def test_fixed_weight_matches_enumeration(n: int, order: Literal[1, 2]) -> None:
    polynomial = _polynomial(n)
    for weight in range(n + 1):
        values = []
        for selected in combinations(range(n), weight):
            ones = set(selected)
            value = sum(
                t.coefficient.as_fraction()
                * (-1) ** len(ones.intersection(t.character))
                for t in polynomial.terms
            )
            values.append(value**order)
        expected = sum(values, Fraction()) / len(values)
        assert fixed_weight_moment(polynomial, weight, order).as_fraction() == expected


def test_large_sphere_remains_compact() -> None:
    polynomial = _polynomial(256)
    mean = fixed_weight_moment(polynomial, 128).as_fraction()
    second = fixed_weight_moment(polynomial, 128, 2).as_fraction()
    assert mean == Fraction(193024, 64515)
    assert second - mean**2 == Fraction(121029545984, 4162185225)


def test_zero_dimensional_sphere_and_zero_polynomial() -> None:
    assert fixed_weight_moment(RationalWalshPolynomial(variable_count=0), 0).num == 0
    polynomial = RationalWalshPolynomial(
        variable_count=0,
        terms=(WalshTerm(character=(), coefficient=CanonicalRational(num=2, den=3)),),
    )
    assert fixed_weight_moment(polynomial, 0, 2).as_fraction() == Fraction(4, 9)


@pytest.mark.parametrize("weight", [-1, 5])
def test_invalid_sphere_weight(weight: int) -> None:
    from jacobian.catalog.models import OperationDomainValidationError

    with pytest.raises(OperationDomainValidationError, match="weight must"):
        fixed_weight_moment(_polynomial(4), weight)


def test_character_square_is_one_at_every_weight() -> None:
    polynomial = RationalWalshPolynomial(
        variable_count=12,
        terms=(
            WalshTerm(
                character=tuple(range(7)), coefficient=CanonicalRational(num=2, den=3)
            ),
        ),
    )
    for weight in range(13):
        assert fixed_weight_moment(polynomial, weight, 2).as_fraction() == Fraction(
            4, 9
        )
