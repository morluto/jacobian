from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.collatz_wielandt.operations import (
    compute_collatz_wielandt_profile,
)


def _cr(num, den=1):
    return CanonicalRational.from_fraction(Fraction(num, den))


def test_identity() -> None:
    matrix = ((_cr(1), _cr(0)), (_cr(0), _cr(1)))
    vector = (_cr(1), _cr(1))
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.quotients[0].as_fraction() == Fraction(1)
    assert result.quotients[1].as_fraction() == Fraction(1)
    assert result.max_quotient.as_fraction() == Fraction(1)


def test_2x2() -> None:
    matrix = ((_cr(2), _cr(1)), (_cr(0), _cr(3)))
    vector = (_cr(1), _cr(1))
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.quotients[0].as_fraction() == Fraction(3)
    assert result.quotients[1].as_fraction() == Fraction(3)


def test_non_uniform_vector() -> None:
    matrix = ((_cr(1), _cr(2)), (_cr(3), _cr(4)))
    vector = (_cr(1), _cr(2))
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.quotients[0].as_fraction() == Fraction(5)
    assert result.quotients[1].as_fraction() == Fraction(11, 2)
    assert result.max_quotient.as_fraction() == Fraction(11, 2)


def test_result_preserves_source() -> None:
    matrix = ((_cr(1),),)
    vector = (_cr(1),)
    result = compute_collatz_wielandt_profile(matrix, vector)
    assert result.matrix == matrix
    assert result.vector == vector
