"""Cyclotomic-factor and complex-modulus slice (#3724)."""

from __future__ import annotations

from fractions import Fraction

from jacobian.math.polynomials._cyclotomic_factor import cyclotomic_factor_profile
from jacobian.math.polynomials._mahler_kernel import (
    mahler_measure,
    quadratic_root_profile,
)
from jacobian.math.polynomials._models import IntegerPolynomial


def test_phi3_identified() -> None:
    result = cyclotomic_factor_profile(IntegerPolynomial(coefficients=(1, 1, 1)))
    assert result.status == "IDENTIFIED_CYCLOTOMIC"
    assert result.cyclotomic_index == 3


def test_phi4_identified() -> None:
    result = cyclotomic_factor_profile(IntegerPolynomial(coefficients=(1, 0, 1)))
    assert result.status == "IDENTIFIED_CYCLOTOMIC"
    assert result.cyclotomic_index == 4


def test_non_cyclotomic_not_identified() -> None:
    result = cyclotomic_factor_profile(IntegerPolynomial(coefficients=(1, 1, -1)))
    assert result.status == "NOT_IDENTIFIED_IN_SUPPORTED_RANGE"
    assert result.cyclotomic_index is None


def test_non_monic_not_identified() -> None:
    result = cyclotomic_factor_profile(IntegerPolynomial(coefficients=(2, 2, 2)))
    assert result.status == "NOT_IDENTIFIED_IN_SUPPORTED_RANGE"


def test_complex_pair_squared_modulus_without_radicals() -> None:
    # x^2 + 1: modulus^2 = 1, ON_UNIT_CIRCLE, Mahler measure 1.
    profile = quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, 1)))
    assert profile.root_kind == "COMPLEX_CONJUGATE"
    assert profile.complex_pair_squared_modulus is not None
    assert profile.complex_pair_squared_modulus.as_fraction() == Fraction(1)
    measure = mahler_measure(IntegerPolynomial(coefficients=(1, 0, 1)))
    assert measure.mahler_measure.as_fraction() == Fraction(1)


def test_reciprocal_quadratic_outside_contribution() -> None:
    # x^2 + 3: modulus^2 = 3, OUTSIDE, Mahler measure 3 via squared modulus.
    profile = quadratic_root_profile(IntegerPolynomial(coefficients=(1, 0, 3)))
    assert profile.root_kind == "COMPLEX_CONJUGATE"
    assert profile.complex_pair_squared_modulus.as_fraction() == Fraction(3)
    assert profile.root_locations == ("OUTSIDE_UNIT_DISK",)
    measure = mahler_measure(IntegerPolynomial(coefficients=(1, 0, 3)))
    assert measure.mahler_measure.as_fraction() == Fraction(3)
