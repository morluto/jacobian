"""Canonical Gaussian-rational scalar tests."""

from fractions import Fraction

from jacobian.math.number_theory.number_fields import GaussianRational


def _gaussian(real: int, imaginary: int) -> GaussianRational:
    return GaussianRational.from_fractions(Fraction(real), Fraction(imaginary))


def test_gaussian_rational_multiplication_uses_i_squared_minus_one() -> None:
    assert _gaussian(1, 1) * _gaussian(1, -1) == _gaussian(2, 0)


def test_gaussian_rational_zero_and_one_round_trip() -> None:
    for value in (GaussianRational.zero(), GaussianRational.one()):
        assert GaussianRational.model_validate_json(value.model_dump_json()) == value
