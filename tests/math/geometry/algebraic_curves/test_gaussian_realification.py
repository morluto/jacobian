"""Tests for Gaussian polynomial realification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.algebraic_curves import (
    gaussian_realification as public_gaussian_realification,
)
from jacobian.math.geometry.algebraic_curves._gaussian_realification import (
    GaussianComplexCoefficient,
    GaussianRealificationRequest,
    UnivariateGaussianPolynomial,
    UnivariateGaussianPolynomialTerm,
    gaussian_realification,
)
from jacobian.math.geometry.algebraic_curves._tools import TOOLS
from jacobian.math.probability import ExactComplexRational


def _cr(real: str, imag: str = "0") -> GaussianComplexCoefficient:
    return GaussianComplexCoefficient(
        real=CanonicalRational(num=real, den="1"),
        imaginary=CanonicalRational(num=imag, den="1"),
    )


def _term(real: str, imag: str, exp: int) -> UnivariateGaussianPolynomialTerm:
    return UnivariateGaussianPolynomialTerm(coefficient=_cr(real, imag), exponent=exp)


def test_catalog_contains_gaussian_realification():
    assert "algebraic_geometry.gaussian_polynomial.realification.compute" in {
        tool.operation_id for tool in TOOLS
    }


def test_native_api_and_gaussian_scalar_compose():
    coefficient = ExactComplexRational(
        real=CanonicalRational(num="1", den="1"),
        imaginary=CanonicalRational(num="2", den="1"),
    )
    polynomial = UnivariateGaussianPolynomial(
        variable="z",
        terms=(UnivariateGaussianPolynomialTerm(coefficient=coefficient, exponent=1),),
    )
    assert (
        public_gaussian_realification(polynomial, ("x", "y")).source_polynomial
        == polynomial
    )


def test_quadratic_gaussian_realification():
    poly = UnivariateGaussianPolynomial(
        variable="z",
        terms=(_term("1", "0", 2), _term("-1", "-1", 0)),
    )
    result = gaussian_realification(poly, ("x", "y"))
    # Expected u = x^2 - y^2 -1, v = 2xy -1
    assert result.real_part.variables == ("x", "y")
    assert result.imag_part.variables == ("x", "y")
    assert result.substitution == "z = x + i*y"
    assert result.method == "BINOMIAL_REALIFICATION"
    real_terms = {
        (t.exponents, t.coefficient.num, t.coefficient.den)
        for t in result.real_part.polynomial.terms
    }
    imag_terms = {
        (t.exponents, t.coefficient.num, t.coefficient.den)
        for t in result.imag_part.polynomial.terms
    }
    assert ((2, 0), "1", "1") in real_terms
    assert ((0, 2), "-1", "1") in real_terms
    assert ((0, 0), "-1", "1") in real_terms
    assert ((1, 1), "2", "1") in imag_terms
    assert ((0, 0), "-1", "1") in imag_terms


def test_real_coefficient_cubic():
    poly = UnivariateGaussianPolynomial(
        variable="z",
        terms=(_term("1", "0", 3), _term("-2", "0", 1), _term("1", "0", 0)),
    )
    result = gaussian_realification(poly, ("x", "y"))
    real_exps = {
        t.exponents: int(t.coefficient.num) for t in result.real_part.polynomial.terms
    }
    imag_exps = {
        t.exponents: int(t.coefficient.num) for t in result.imag_part.polynomial.terms
    }
    assert real_exps[(3, 0)] == 1
    assert real_exps[(1, 2)] == -3
    assert real_exps[(1, 0)] == -2
    assert real_exps[(0, 0)] == 1
    assert imag_exps[(2, 1)] == 3
    assert imag_exps[(0, 3)] == -1
    assert imag_exps[(0, 1)] == -2


def test_zero_polynomial():
    poly = UnivariateGaussianPolynomial(variable="z", terms=())
    result = gaussian_realification(poly, ("x", "y"))
    assert result.real_part.polynomial.terms == ()
    assert result.imag_part.polynomial.terms == ()
    assert result.real_part.variables == ("x", "y")


def test_purely_imaginary_constant():
    poly = UnivariateGaussianPolynomial(variable="z", terms=(_term("0", "1", 0),))
    result = gaussian_realification(poly, ("x", "y"))
    assert result.real_part.polynomial.terms == ()
    assert len(result.imag_part.polynomial.terms) == 1
    assert result.imag_part.polynomial.terms[0].exponents == (0, 0)
    assert result.imag_part.polynomial.terms[0].coefficient.num == "1"


def test_monomial_i_z_squared():
    poly = UnivariateGaussianPolynomial(variable="z", terms=(_term("0", "1", 2),))
    result = gaussian_realification(poly, ("x", "y"))
    # i*z^2 => i*(x^2 - y^2 +2i xy)= i(x^2 - y^2) -2xy => real -2xy, imag x^2 - y^2
    real = {
        (t.exponents): int(t.coefficient.num) for t in result.real_part.polynomial.terms
    }
    imag = {
        (t.exponents): int(t.coefficient.num) for t in result.imag_part.polynomial.terms
    }
    assert real[(1, 1)] == -2
    assert imag[(2, 0)] == 1
    assert imag[(0, 2)] == -1


def test_target_labels_change_only_axis():
    poly = UnivariateGaussianPolynomial(variable="z", terms=(_term("1", "0", 1),))
    result_xy = gaussian_realification(poly, ("x", "y"))
    result_uv = gaussian_realification(poly, ("u", "v"))
    assert result_xy.real_part.polynomial.terms[0].exponents == (1, 0)
    assert result_uv.real_part.polynomial.terms[0].exponents == (1, 0)
    assert result_xy.real_part.variables == ("x", "y")
    assert result_uv.real_part.variables == ("u", "v")
    assert result_uv.substitution == "z = u + i*v"


def test_rejects_target_collision_with_source():
    poly = UnivariateGaussianPolynomial(variable="z", terms=(_term("1", "0", 1),))
    with pytest.raises(ValidationError, match="distinct from the source"):
        GaussianRealificationRequest(polynomial=poly, target_variables=("z", "y"))
    with pytest.raises(ValidationError, match="distinct"):
        GaussianRealificationRequest(polynomial=poly, target_variables=("x", "x"))


def test_defining_invariant_reconstruction():
    # For every returned pair, check reconstruction via binomial expansion replay
    from fractions import Fraction

    poly = UnivariateGaussianPolynomial(
        variable="z",
        terms=(_term("2", "1", 2), _term("-1", "0", 1), _term("3", "-2", 0)),
    )
    result = gaussian_realification(poly, ("x", "y"))
    # Replay expansion independently and compare dictionaries
    from math import comb

    real_ref: dict[tuple[int, int], Fraction] = {}
    imag_ref: dict[tuple[int, int], Fraction] = {}
    for term in poly.terms:
        k = term.exponent
        a_real, a_imag = term.coefficient.as_fractions()
        for j in range(k + 1):
            coeff = Fraction(comb(k, j), 1)
            exp = (k - j, j)
            ij = j % 4
            if ij == 0:
                r, i = a_real, a_imag
            elif ij == 1:
                r, i = -a_imag, a_real
            elif ij == 2:
                r, i = -a_real, -a_imag
            else:
                r, i = a_imag, -a_real
            r_contrib = coeff * r
            i_contrib = coeff * i
            if r_contrib != 0:
                real_ref[exp] = real_ref.get(exp, Fraction(0, 1)) + r_contrib
            if i_contrib != 0:
                imag_ref[exp] = imag_ref.get(exp, Fraction(0, 1)) + i_contrib
    real_actual = {
        t.exponents: t.coefficient.as_fraction()
        for t in result.real_part.polynomial.terms
    }
    imag_actual = {
        t.exponents: t.coefficient.as_fraction()
        for t in result.imag_part.polynomial.terms
    }
    # Filter zeros
    real_ref = {k: v for k, v in real_ref.items() if v != 0}
    imag_ref = {k: v for k, v in imag_ref.items() if v != 0}
    assert real_actual == real_ref
    assert imag_actual == imag_ref


def test_json_round_trip():
    poly = UnivariateGaussianPolynomial(variable="z", terms=(_term("1", "0", 1),))
    result = gaussian_realification(poly, ("x", "y"))
    json_val = result.model_dump_json()
    replay = type(result).model_validate_json(json_val, strict=True)
    assert replay == result


def test_admission_rejects_excessive_degree():
    with pytest.raises(ValidationError, match=r"64|less_than_equal|degree"):
        UnivariateGaussianPolynomial(
            variable="z",
            terms=(
                UnivariateGaussianPolynomialTerm(coefficient=_cr("1"), exponent=65),
            ),
        )


def test_admission_bounds_each_component_before_expansion():
    polynomial = UnivariateGaussianPolynomial(
        variable="z",
        terms=tuple(_term("1", "1", degree) for degree in range(64, 59, -1)),
    )
    with pytest.raises(Exception, match="realification component"):
        gaussian_realification(polynomial, ("x", "y"))


def test_admission_reserves_binomial_coefficient_digits():
    polynomial = UnivariateGaussianPolynomial(
        variable="z",
        terms=(_term("9" * 256, "0", 64),),
    )
    with pytest.raises(Exception, match="Gaussian coefficient"):
        gaussian_realification(polynomial, ("x", "y"))
