"""Tests for bounded exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

import copy
from copy import deepcopy
from fractions import Fraction
from typing import Any

import pytest
from tests.math.polynomials._support import polynomial_validation_error

from jacobian.math.polynomials.multivariate._division import (
    MultivariateDivisionRequest,
)
from jacobian.math.polynomials.multivariate._gcd import MultivariateGcdRequest
from jacobian.math.polynomials.multivariate._models import (
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
)
from jacobian.math.polynomials.multivariate._operations import (
    compute_multivariate_division,
    compute_multivariate_gcd,
    compute_multivariate_resultant,
    compute_multivariate_subresultant_sequence,
)
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantRequest,
    MultivariateResultantResult,
    _verify_multivariate_resultant,
)
from jacobian.math.polynomials.multivariate._subresultants import (
    MultivariateSubresultantSequenceRequest,
    MultivariateSubresultantSequenceResult,
    _verify_multivariate_subresultant_sequence,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...],
    terms: tuple[tuple[str, tuple[int, ...]], ...],
) -> RationalPolynomial:
    """Build a ``RationalPolynomial`` from ``"num/den", (exponents,)`` tuples."""

    return RationalPolynomial.model_validate(
        {
            "domain": "QQ",
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": num, "den": den}, "exponents": list(exp)}
                    for coeff, exp in terms
                    for num, den in [coeff.split("/")]
                ]
            },
        }
    )


def _poly_from_sympy(expr) -> RationalPolynomial:
    """Convert a small two-variable SymPy expression to a ``RationalPolynomial``."""

    from sympy import Poly, symbols

    x_symbol, y_symbol = symbols("x y")
    poly = Poly(expr, x_symbol, y_symbol)
    terms = []
    for monomials, coefficient in sorted(poly.terms(), reverse=True):
        terms.append((f"{coefficient.p}/{coefficient.q}", tuple(monomials)))
    return _poly(("x", "y"), tuple(terms))


def _scalar(
    variables: tuple[str, ...],
    value: str,
) -> RationalPolynomial:
    """Build a scalar (constant) polynomial for the given variable ring."""

    num, _, den = value.partition("/")
    if not den:
        den = "1"
    return _poly(variables, ((f"{num}/{den}", (0,) * len(variables)),))


# --------------------------------------------------------------------------- #
# Multivariate GCD
# --------------------------------------------------------------------------- #


class TestMultivariateGcd:
    """Tests for ``polynomial.multivariate.gcd.compute``."""

    def test_gcd_of_non_coprime_pair(self) -> None:
        """gcd(x^2*y, x*y) = x*y."""

        left = _poly(("x", "y"), (("1/1", (2, 1)),))
        right = _poly(("x", "y"), (("1/1", (1, 1)),))
        result = compute_multivariate_gcd(
            MultivariateGcdRequest(left=left, right=right)
        )
        gcd = result.gcd
        assert gcd.variables == ("x", "y")
        assert len(gcd.polynomial.terms) == 1
        term = gcd.polynomial.terms[0]
        assert term.coefficient.num == "1"
        assert term.coefficient.den == "1"
        assert term.exponents == (1, 1)

    def test_gcd_of_coprime_pair(self) -> None:
        """gcd(x*y - 1, x^2 - 1) = 1 (coprime over QQ[x, y])."""

        left = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        right = _poly(("x", "y"), (("1/1", (2, 0)), ("-1/1", (0, 0))))
        result = compute_multivariate_gcd(
            MultivariateGcdRequest(left=left, right=right)
        )
        gcd = result.gcd
        assert len(gcd.polynomial.terms) == 1
        term = gcd.polynomial.terms[0]
        assert term.coefficient.num == "1"
        assert term.coefficient.den == "1"
        assert term.exponents == (0, 0)

    def test_gcd_is_monic(self) -> None:
        """The GCD should be normalized to a monic (leading-coefficient 1) polynomial."""

        # gcd(2*x*y, 3*x*y) = x*y (content stripped, monic associate)
        left = _poly(("x", "y"), (("2/1", (1, 1)),))
        right = _poly(("x", "y"), (("3/1", (1, 1)),))
        result = compute_multivariate_gcd(
            MultivariateGcdRequest(left=left, right=right)
        )
        gcd = result.gcd
        assert len(gcd.polynomial.terms) == 1
        term = gcd.polynomial.terms[0]
        assert term.coefficient.num == "1"
        assert term.coefficient.den == "1"

    def test_gcd_with_rational_coefficients(self) -> None:
        """GCD works over QQ with non-integer coefficients."""

        left = _poly(("x", "y"), (("1/2", (2, 1)),))
        right = _poly(("x", "y"), (("1/3", (1, 1)),))
        result = compute_multivariate_gcd(
            MultivariateGcdRequest(left=left, right=right)
        )
        gcd = result.gcd
        # gcd(x^2*y, x*y) = x*y, monic
        assert len(gcd.polynomial.terms) == 1
        term = gcd.polynomial.terms[0]
        assert term.coefficient.num == "1"
        assert term.coefficient.den == "1"

    def test_gcd_rejects_univariate(self) -> None:
        """Univariate polynomials are rejected for multivariate operations."""

        left = _poly(("x",), (("1/1", (1,)),))
        right = _poly(("x",), (("1/1", (0,)),))
        with pytest.raises(ValueError, match="at least two variables"):
            MultivariateGcdRequest(left=left, right=right)

    def test_gcd_rejects_mismatched_variables(self) -> None:
        """Polynomials must share the same ordered variable list."""

        left = _poly(("x", "y"), (("1/1", (1, 1)),))
        right = _poly(("x", "z"), (("1/1", (1, 1)),))
        with pytest.raises(ValueError, match="same ordered variables"):
            MultivariateGcdRequest(left=left, right=right)

    def test_gcd_rejects_oversized_terms(self) -> None:
        """Operations fail closed on oversized inputs."""

        # Build >512 unique monomials (i, 0) for i = 599..0 (descending order).
        terms = tuple((f"{i + 1}/1", (i, 0)) for i in range(599, -1, -1))
        left = _poly(("x", "y"), terms[:600])
        right = _poly(("x", "y"), (("1/1", (0, 0)),))
        with pytest.raises(ValueError, match="term operation budget"):
            MultivariateGcdRequest(left=left, right=right)


# --------------------------------------------------------------------------- #
# Multivariate division
# --------------------------------------------------------------------------- #


class TestMultivariateDivision:
    """Tests for ``polynomial.multivariate.divide.compute``."""

    def test_rejects_zero_divisor(self) -> None:

        left = _poly(("x", "y"), (("1/1", (1, 0)),))
        right = _poly(("x", "y"), ())
        with polynomial_validation_error():
            MultivariateDivisionRequest(left=left, right=right)

    def test_division_exact(self) -> None:
        """x^2*y / (x*y) = x, remainder 0."""

        left = _poly(("x", "y"), (("1/1", (2, 1)),))
        right = _poly(("x", "y"), (("1/1", (1, 1)),))
        result = compute_multivariate_division(
            MultivariateDivisionRequest(left=left, right=right)
        )
        assert len(result.remainder.polynomial.terms) == 0
        quotient = result.quotient
        assert len(quotient.polynomial.terms) == 1
        assert quotient.polynomial.terms[0].exponents == (1, 0)
        assert quotient.polynomial.terms[0].coefficient.num == "1"

    def test_division_with_remainder(self) -> None:
        """Divide x^2*y + x by x*y - 1: quotient = x, remainder = 2*x."""

        left = _poly(("x", "y"), (("1/1", (2, 1)), ("1/1", (1, 0))))
        right = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        result = compute_multivariate_division(
            MultivariateDivisionRequest(left=left, right=right, monomial_order="lex")
        )
        quotient = result.quotient
        remainder = result.remainder
        assert len(quotient.polynomial.terms) == 1
        assert quotient.polynomial.terms[0].exponents == (1, 0)
        assert quotient.polynomial.terms[0].coefficient.num == "1"
        assert len(remainder.polynomial.terms) == 1
        assert remainder.polynomial.terms[0].exponents == (1, 0)
        assert remainder.polynomial.terms[0].coefficient.num == "2"

    def test_division_grlex_order(self) -> None:
        """Division under grlex order should be a valid reconstruction."""

        left = _poly(
            ("x", "y"),
            (("1/1", (2, 1)), ("1/1", (1, 0))),
        )
        right = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        result = compute_multivariate_division(
            MultivariateDivisionRequest(left=left, right=right, monomial_order="grlex")
        )
        assert result.monomial_order == "grlex"

    def test_division_grevlex_order(self) -> None:
        """Division under grevlex order should be a valid reconstruction."""

        left = _poly(
            ("x", "y"),
            (("1/1", (2, 1)), ("1/1", (1, 0))),
        )
        right = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        result = compute_multivariate_division(
            MultivariateDivisionRequest(
                left=left, right=right, monomial_order="grevlex"
            )
        )
        assert result.monomial_order == "grevlex"

    def test_division_zero_dividend(self) -> None:
        """Dividing the zero polynomial gives zero quotient and remainder."""

        left = _poly(("x", "y"), ())
        right = _poly(("x", "y"), (("1/1", (1, 1)),))
        result = compute_multivariate_division(
            MultivariateDivisionRequest(left=left, right=right)
        )
        assert len(result.quotient.polynomial.terms) == 0
        assert len(result.remainder.polynomial.terms) == 0

    def test_division_rejects_univariate(self) -> None:
        """Univariate polynomials are rejected for multivariate operations."""

        left = _poly(("x",), (("1/1", (2,)),))
        right = _poly(("x",), (("1/1", (1,)),))
        with pytest.raises(ValueError, match="at least two variables"):
            MultivariateDivisionRequest(left=left, right=right)

    def test_division_rejects_mismatched_variables(self) -> None:
        """Polynomials must share the same ordered variable list."""

        left = _poly(("x", "y"), (("1/1", (1, 1)),))
        right = _poly(("a", "b"), (("1/1", (1, 1)),))
        with pytest.raises(ValueError, match="same ordered variables"):
            MultivariateDivisionRequest(left=left, right=right)


# --------------------------------------------------------------------------- #
# Multivariate resultant
# --------------------------------------------------------------------------- #


class TestMultivariateResultant:
    """Tests for ``polynomial.multivariate.resultant.compute``."""

    def test_resultant_bivariate(self) -> None:
        """res(x*y - 1, x^2 - 1, x) = 1 - y^2 (a polynomial in y)."""

        left = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        right = _poly(("x", "y"), (("1/1", (2, 0)), ("-1/1", (0, 0))))
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )
        )
        assert result.elimination_variable == "x"
        # Resultant should be a polynomial in the remaining variable y.
        assert result.resultant.kind == "POLYNOMIAL"
        value = result.resultant.value
        assert value.variables == ("y",)
        # The resultant is 1 - y^2, i.e. terms {y^2: -1, y^0: 1}.
        terms = {
            t.exponents[0]: (t.coefficient.num, t.coefficient.den)
            for t in value.polynomial.terms
        }
        assert terms.get(2) == ("-1", "1")
        assert terms.get(0) == ("1", "1")

    def test_resultant_eliminating_different_variable(self) -> None:
        """res(x*y - 1, y^2 - x, y) = 1 - x^3 (a polynomial in x)."""

        left = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        right = _poly(("x", "y"), (("-1/1", (1, 0)), ("1/1", (0, 2))))
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="y"
            )
        )
        assert result.elimination_variable == "y"
        assert result.resultant.kind == "POLYNOMIAL"
        value = result.resultant.value
        assert value.variables == ("x",)

    def test_resultant_with_three_variables(self) -> None:
        """res(x^2 - y, x - z, x) = z^2 - y (a polynomial in y, z)."""

        left = _poly(("x", "y", "z"), (("1/1", (2, 0, 0)), ("-1/1", (0, 1, 0))))
        right = _poly(("x", "y", "z"), (("1/1", (1, 0, 0)), ("-1/1", (0, 0, 1))))
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )
        )
        assert result.elimination_variable == "x"
        assert result.resultant.kind == "POLYNOMIAL"
        value = result.resultant.value
        assert value.variables == ("y", "z")

    def test_resultant_nonzero_constant_right_input(self) -> None:
        """Atlas elimination case: res_x(x + y^2 - u, y - v) = y - v."""

        left = _poly(
            ("x", "y", "u", "v"),
            (("1/1", (1, 0, 0, 0)), ("1/1", (0, 2, 0, 0)), ("-1/1", (0, 0, 1, 0))),
        )
        right = _poly(
            ("x", "y", "u", "v"), (("1/1", (0, 1, 0, 0)), ("-1/1", (0, 0, 0, 1)))
        )
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )
        )
        assert result.resultant.kind == "POLYNOMIAL"
        terms = {
            t.exponents: t.coefficient.as_fraction()
            for t in result.resultant.value.polynomial.terms
        }
        assert terms == {(1, 0, 0): Fraction(1), (0, 0, 1): Fraction(-1)}

    def test_resultant_power_rule_over_constant_inputs(self) -> None:
        """Res_x(f, c) = c^deg_x(f) and Res_x(c, g) = c^deg_x(g)."""

        f = _poly(("x", "y"), (("3/1", (3, 0)), ("1/2", (0, 1))))
        five = _poly(("x", "y"), (("5/1", (0, 0)),))
        left_as_eliminated = compute_multivariate_resultant(
            MultivariateResultantRequest(left=f, right=five, elimination_variable="x")
        )
        right_as_eliminated = compute_multivariate_resultant(
            MultivariateResultantRequest(left=five, right=f, elimination_variable="x")
        )
        for record in (left_as_eliminated, right_as_eliminated):
            assert record.resultant.kind == "POLYNOMIAL"
            terms = {
                t.exponents: t.coefficient.as_fraction()
                for t in record.resultant.value.polynomial.terms
            }
            assert terms == {(0,): Fraction(125)}

    def test_resultant_both_constant_gives_empty_determinant_value(self) -> None:
        """Two inputs constant in the eliminated variable give exactly 1."""

        left = _poly(("x", "y", "z"), (("1/2", (0, 1, 0)), ("5/1", (0, 0, 0))))
        right = _poly(("x", "y", "z"), (("1/3", (0, 0, 1)), ("-7/1", (0, 0, 0))))
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )
        )
        assert result.resultant.kind == "POLYNOMIAL"
        terms = {
            t.exponents: t.coefficient.as_fraction()
            for t in result.resultant.value.polynomial.terms
        }
        assert terms == {(0, 0): Fraction(1)}

    def test_resultant_zero_input_gives_zero(self) -> None:
        """Either zero input gives the zero resultant."""

        zero = _poly(("x", "y"), ())
        g = _poly(("x", "y"), (("1/1", (2, 0)), ("-4/1", (0, 1))))
        for left, right in ((zero, g), (g, zero), (zero, zero)):
            result = compute_multivariate_resultant(
                MultivariateResultantRequest(
                    left=left, right=right, elimination_variable="x"
                )
            )
            assert result.resultant.kind == "POLYNOMIAL"
            assert len(result.resultant.value.polynomial.terms) == 0

    def test_resultant_swap_law_with_degenerate_rows(self) -> None:
        """Res(f, g) = (-1)^{mn} Res(g, f) across admitted degree profiles."""

        cases = (
            (
                _poly(("x", "y"), (("1/1", (2, 0)), ("-3/1", (0, 0)))),
                _poly(("x", "y"), (("1/1", (1, 1)), ("2/1", (0, 0)))),
            ),
            (
                _poly(("x", "y"), (("1/1", (1, 0)),)),
                _poly(("x", "y"), (("7/1", (0, 0)),)),
            ),
            (
                _poly(("x", "y"), (("1/1", (0, 1)),)),
                _poly(("x", "y"), (("1/1", (3, 0)), ("-1/1", (0, 0)))),
            ),
            (
                _poly(("x", "y"), (("1/1", (1, 0)), ("5/1", (0, 0)))),
                _poly(("x", "y"), (("1/1", (3, 0)), ("-4/1", (0, 0)))),
            ),
        )
        for left, right in cases:
            first = compute_multivariate_resultant(
                MultivariateResultantRequest(
                    left=left, right=right, elimination_variable="x"
                )
            )
            second = compute_multivariate_resultant(
                MultivariateResultantRequest(
                    left=right, right=left, elimination_variable="x"
                )
            )
            m = max((t.exponents[0] for t in left.polynomial.terms), default=0)
            n = max((t.exponents[0] for t in right.polynomial.terms), default=0)
            sign = -1 if (m * n) % 2 else 1
            first_terms = {
                t.exponents: sign * t.coefficient.as_fraction()
                for t in first.resultant.value.polynomial.terms
            }
            second_terms = {
                t.exponents: t.coefficient.as_fraction()
                for t in second.resultant.value.polynomial.terms
            }
            assert first_terms == second_terms

    def test_resultant_sign_matches_sylvester_orientation(self) -> None:
        """deg(left) < deg(right) keeps the standard Sylvester orientation.

        SymPy's PRS canonicalizes the degree order without compensating the
        swap sign (upstream sympy/sympy#10666); these fixed cases flip under
        the uncorrected backend value.
        """

        left = _poly(("x", "y"), (("1/1", (1, 0)), ("5/1", (0, 0))))
        right = _poly(("x", "y"), (("1/1", (3, 0)),))
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )
        )
        terms = {
            t.exponents: t.coefficient.as_fraction()
            for t in result.resultant.value.polynomial.terms
        }
        assert terms == {(0,): Fraction(-125)}

        symbolic_left = _poly(("x", "y"), (("1/1", (1, 1)),))
        symbolic_right = _poly(("x", "y"), (("1/1", (3, 0)), ("-1/1", (0, 3))))
        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=symbolic_left, right=symbolic_right, elimination_variable="x"
            )
        )
        terms = {
            t.exponents: t.coefficient.as_fraction()
            for t in result.resultant.value.polynomial.terms
        }
        assert terms == {(6,): Fraction(-1)}

    def test_resultant_matches_sylvester_determinant_oracle(self) -> None:
        """Differential oracle: the sparse value equals the Sylvester determinant."""

        from sympy import Matrix, symbols, together
        from sympy.polys.subresultants_qq_zz import sylvester as sympy_sylvester

        x, y = symbols("x y")
        cases = (
            (x**2 + y * x + 1, 3 * x - y**2),
            (x + 5, x**3 + y),
            (2 * x**2 - 7, y * x + 3),
            (y**2 + 1, x - 4),
        )
        for f_expr, g_expr in cases:
            f = _poly_from_sympy(f_expr)
            g = _poly_from_sympy(g_expr)
            result = compute_multivariate_resultant(
                MultivariateResultantRequest(left=f, right=g, elimination_variable="x")
            )
            expected = together(Matrix(sympy_sylvester(f_expr, g_expr, x)).det())
            claimed = sum(
                Fraction(int(t.coefficient.num), int(t.coefficient.den))
                * y ** t.exponents[0]
                for t in result.resultant.value.polynomial.terms
            )
            assert together(expected - claimed) == 0

    def test_resultant_parsing_is_structural_and_verification_is_deliberate(
        self,
    ) -> None:
        """Deserialization does not replay the determinant of retained sources."""

        left = _poly(("x", "y"), (("1/1", (1, 1)), ("-1/1", (0, 0))))
        right = _poly(("x", "y"), (("1/1", (2, 0)), ("-1/1", (0, 0))))
        request = MultivariateResultantRequest(
            left=left, right=right, elimination_variable="x"
        )
        result = compute_multivariate_resultant(request)
        dumped = result.model_dump()
        assert MultivariateResultantResult.model_validate(dumped) == result
        assert _verify_multivariate_resultant(result)

        forged = copy.deepcopy(dumped)
        forged["resultant"]["value"]["polynomial"]["terms"] = [
            {"coefficient": {"num": "9999", "den": "1"}, "exponents": [0]}
        ]
        forged_result = MultivariateResultantResult.model_validate(forged)
        assert not _verify_multivariate_resultant(forged_result)

        swapped = copy.deepcopy(dumped)
        swapped["left"] = dumped["right"]
        swapped_result = MultivariateResultantResult.model_validate(swapped)
        assert not _verify_multivariate_resultant(swapped_result)

        out_of_envelope = copy.deepcopy(dumped)
        out_of_envelope["left"]["polynomial"]["terms"][0]["exponents"] = (100, 1)
        parsed_out_of_envelope = MultivariateResultantResult.model_validate(
            out_of_envelope
        )
        assert not _verify_multivariate_resultant(parsed_out_of_envelope)

    def test_resultant_rejects_univariate(self) -> None:
        """Univariate polynomials are rejected for multivariate operations."""

        left = _poly(("x",), (("1/1", (2,)), ("-1/1", (0,))))
        right = _poly(("x",), (("1/1", (1,)), ("-2/1", (0,))))
        with pytest.raises(ValueError, match="at least two variables"):
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )

    def test_resultant_rejects_variable_not_in_ring(self) -> None:
        """The elimination variable must belong to the declared ring."""

        left = _poly(("x", "y"), (("1/1", (1, 1)),))
        right = _poly(("x", "y"), (("1/1", (2, 0)),))
        with pytest.raises(ValueError, match="must belong to the declared ring"):
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="z"
            )

    def test_resultant_rejects_oversized_degree(self) -> None:
        """Operations fail closed on oversized inputs."""

        # Build polynomials with degree sum > 64 in the elimination variable.
        # Terms must be in descending lex order.
        terms_left = tuple(("1/1", (i, 0)) for i in range(39, -1, -1))
        terms_right = tuple(("1/1", (i, 0)) for i in range(39, -1, -1))
        left = _poly(("x", "y"), terms_left)
        right = _poly(("x", "y"), terms_right)
        with pytest.raises(ValueError, match="Sylvester degree"):
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )

    def test_resultant_rejects_unbounded_remaining_variable_expansion(self) -> None:
        """Reject a resultant whose possible monomial support exceeds its output budget."""

        variables = ("x", "y1", "y2", "y3", "y4", "y5", "y6", "y7")
        zeroes = (0,) * len(variables)
        left_terms = [("1/1", (2, *zeroes[1:]))]
        left_terms.extend(
            ("-1/1", (0, *(1 if index == offset else 0 for index in range(7))))
            for offset in range(7)
        )
        right_terms = [("1/1", (31, *zeroes[1:])), ("-1/1", zeroes)]
        left = _poly(variables, tuple(left_terms))
        right = _poly(variables, tuple(right_terms))
        with pytest.raises(ValueError, match="resultant output"):
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )

    def test_resultant_preserves_unequal_degree_source_orientation(self) -> None:
        """res(x+y, x^3+1, x) = 1-y^3, not the swapped-input sign."""

        left = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("1/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (3, 0)), ("1/1", (0, 0))),
        )

        result = compute_multivariate_resultant(
            MultivariateResultantRequest(
                left=left,
                right=right,
                elimination_variable="x",
            )
        )

        assert result.resultant.kind == "POLYNOMIAL"
        assert result.resultant.value == _poly(
            ("y",),
            (("-1/1", (3,)), ("1/1", (0,))),
        )


# --------------------------------------------------------------------------- #
# Multivariate-coefficient subresultant sequences
# --------------------------------------------------------------------------- #


class TestMultivariateSubresultantSequence:
    """Tests for ``polynomial.multivariate.subresultant_sequence.compute``."""

    def test_returns_exact_members_lift_and_terminal_resultant(self) -> None:
        """The degree-one member retains the exact lifting relation x-y."""

        left = _poly(
            ("x", "y"),
            (("1/1", (2, 0)), ("-1/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("-1/1", (0, 1))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )

        assert result.convention == "BROWN_SUBRESULTANT_PRS"
        assert result.source_order == "LEFT_RIGHT"
        assert tuple(member.degree_in_main_variable for member in result.members) == (
            2,
            1,
            0,
        )
        assert tuple(member.polynomial for member in result.members) == (
            left,
            right,
            _poly(
                ("x", "y"),
                (("1/1", (0, 2)), ("-1/1", (0, 1))),
            ),
        )
        assert result.skipped_member_degrees == ()
        assert result.resultant == _poly(
            ("y",),
            (("1/1", (2,)), ("-1/1", (1,))),
        )
        assert tuple(
            coefficient.index
            for coefficient in result.principal_subresultant_coefficients
        ) == (0, 1)
        assert tuple(
            coefficient.coefficient
            for coefficient in result.principal_subresultant_coefficients
        ) == (result.resultant, _scalar(("y",), "1"))
        assert result.gcd_member_index == 2
        assert result.gcd_degree_in_main_variable == 0
        assert result.gcd_member_leading_coefficient == result.resultant

    def test_distinguishes_zero_principal_coefficient_from_skipped_member(self) -> None:
        """An abnormal degree drop retains the vanishing formal coefficient."""

        left = _poly(
            ("x", "y"),
            (("1/1", (2, 0)), ("1/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (2, 0)), ("-1/1", (0, 1))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )

        assert tuple(member.degree_in_main_variable for member in result.members) == (
            2,
            2,
            0,
        )
        assert result.members[-1].polynomial == _poly(
            ("x", "y"),
            (("-2/1", (0, 1)),),
        )
        assert result.skipped_member_degrees == (1,)
        assert tuple(
            coefficient.index
            for coefficient in result.principal_subresultant_coefficients
        ) == (0, 1, 2)
        assert tuple(
            coefficient.coefficient
            for coefficient in result.principal_subresultant_coefficients
        ) == (
            _poly(("y",), (("4/1", (2,)),)),
            _poly(("y",), ()),
            _scalar(("y",), "1"),
        )
        assert result.resultant == _poly(("y",), (("4/1", (2,)),))
        assert (
            MultivariateSubresultantSequenceResult.model_validate(
                result.model_dump(mode="json")
            )
            == result
        )

    def test_records_reordering_sign_and_skipped_degree(self) -> None:
        """Unequal source degrees expose both PRS order and resultant sign."""

        linear = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("1/1", (0, 1))),
        )
        cubic = _poly(
            ("x", "y"),
            (("1/1", (3, 0)), ("1/1", (0, 0))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=linear,
                right=cubic,
                main_variable="x",
            )
        )
        swapped = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=cubic,
                right=linear,
                main_variable="x",
            )
        )

        assert result.source_order == "RIGHT_LEFT"
        assert result.resultant_sign_from_sequence_order == -1
        assert result.skipped_member_degrees == (2,)
        assert tuple(member.polynomial for member in result.members) == tuple(
            member.polynomial for member in swapped.members
        )
        assert result.resultant == _poly(
            ("y",),
            (("-1/1", (3,)), ("1/1", (0,))),
        )
        assert result.principal_subresultant_coefficients[0].coefficient == _poly(
            ("y",),
            (("1/1", (3,)), ("-1/1", (0,))),
        )
        assert swapped.resultant == _poly(
            ("y",),
            (("1/1", (3,)), ("-1/1", (0,))),
        )

    def test_records_nonconsecutive_degree_drops(self) -> None:
        left = _poly(
            ("x", "y"),
            (("1/1", (4, 0)), ("1/1", (1, 1)), ("1/1", (0, 0))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (2, 0)), ("1/1", (0, 1))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )

        assert tuple(member.degree_in_main_variable for member in result.members) == (
            4,
            2,
            1,
            0,
        )
        assert result.skipped_member_degrees == (3,)
        assert tuple(
            coefficient.index
            for coefficient in result.principal_subresultant_coefficients
        ) == (0, 1, 2)

    def test_identifies_fraction_field_gcd_and_its_unit(self) -> None:
        """The last member is an explicit polynomial-ring associate of x-y."""

        left = _poly(
            ("x", "y"),
            (
                ("1/1", (2, 0)),
                ("-1/1", (1, 1)),
                ("1/1", (1, 0)),
                ("-1/1", (0, 1)),
            ),
        )
        right = _poly(
            ("x", "y"),
            (
                ("1/1", (3, 0)),
                ("-1/1", (2, 1)),
                ("1/1", (1, 1)),
                ("-1/1", (0, 2)),
            ),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )

        assert result.resultant == _poly(("y",), ())
        assert result.gcd_degree_in_main_variable == 1
        assert result.gcd_member_leading_coefficient == _poly(
            ("y",),
            (("1/1", (1,)), ("1/1", (0,))),
        )
        assert result.members[-1].polynomial == _poly(
            ("x", "y"),
            (
                ("1/1", (1, 1)),
                ("1/1", (1, 0)),
                ("-1/1", (0, 2)),
                ("-1/1", (0, 1)),
            ),
        )

    def test_main_variable_is_an_explicit_axis_choice(self) -> None:
        left = _poly(
            ("x", "y"),
            (("1/1", (2, 0)), ("-1/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("-1/1", (0, 1))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="y",
            )
        )

        assert tuple(member.degree_in_main_variable for member in result.members) == (
            1,
            1,
            0,
        )
        assert result.resultant == _poly(
            ("x",),
            (("1/1", (2,)), ("-1/1", (1,))),
        )

    def test_coherent_variable_renaming_and_reordering_preserves_result(self) -> None:
        """Rename y to z and move the main variable x to the second axis."""

        left = _poly(
            ("z", "u"),
            (("-1/1", (1, 0)), ("1/1", (0, 2))),
        )
        right = _poly(
            ("z", "u"),
            (("-1/1", (1, 0)), ("1/1", (0, 1))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="u",
            )
        )

        assert tuple(member.degree_in_main_variable for member in result.members) == (
            2,
            1,
            0,
        )
        assert result.resultant == _poly(
            ("z",),
            (("1/1", (2,)), ("-1/1", (1,))),
        )

    def test_rational_unit_rescaling_has_exact_resultant_weight(self) -> None:
        left = _poly(
            ("x", "y"),
            (("2/1", (2, 0)), ("-2/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("3/1", (1, 0)), ("-3/1", (0, 1))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )

        assert result.resultant == _poly(
            ("y",),
            (("18/1", (2,)), ("-18/1", (1,))),
        )

    def test_rejects_zero_main_variable_degree(self) -> None:
        constant_in_x = _poly(("x", "y"), (("1/1", (0, 1)),))
        positive_degree = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("1/1", (0, 0))),
        )

        with pytest.raises(ValueError, match="positive main-variable degree"):
            MultivariateSubresultantSequenceRequest(
                left=constant_in_x,
                right=positive_degree,
                main_variable="x",
            )

    def test_rejects_mismatched_ordered_coefficient_rings(self) -> None:
        left = _poly(("x", "y"), (("1/1", (1, 0)),))
        right = _poly(("x", "z"), (("1/1", (1, 0)),))

        with pytest.raises(ValueError, match="same ordered variables"):
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )

    def test_accepts_exact_input_coefficient_boundary(self) -> None:
        boundary_coefficient = "9" * _MAX_MULTIVARIATE_COEFFICIENT_DIGITS
        left = _poly(
            ("x", "y"),
            ((f"{boundary_coefficient}/1", (1, 0)), ("1/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("1/1", (0, 0))),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )
        assert result.resultant == _poly(
            ("y",),
            (("-1/1", (1,)), (f"{boundary_coefficient}/1", (0,))),
        )

        beyond = _poly(
            ("x", "y"),
            (
                (f"{'9' * (_MAX_MULTIVARIATE_COEFFICIENT_DIGITS + 1)}/1", (1, 0)),
                ("1/1", (0, 1)),
            ),
        )
        with pytest.raises(
            ValueError, match=rf"{_MAX_MULTIVARIATE_COEFFICIENT_DIGITS}-digit bound"
        ):
            MultivariateSubresultantSequenceRequest(
                left=beyond,
                right=right,
                main_variable="x",
            )

    def test_admits_through_derived_brown_intermediate_boundary(self) -> None:
        boundary_left = _poly(
            ("x", "y"),
            (("1/1", (18, 0)), ("1/1", (0, 0))),
        )
        boundary_right = _poly(
            ("x", "y"),
            (("1/1", (18, 0)), ("2/1", (0, 0))),
        )
        boundary_result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=boundary_left,
                right=boundary_right,
                main_variable="x",
            )
        )
        assert tuple(
            member.degree_in_main_variable for member in boundary_result.members
        ) == (18, 18, 0)

        beyond = _poly(
            ("x", "y"),
            (("1/1", (19, 0)), ("1/1", (0, 0))),
        )
        with pytest.raises(ValueError, match="Brown pseudo-remainder intermediate"):
            MultivariateSubresultantSequenceRequest(
                left=beyond,
                right=boundary_right,
                main_variable="x",
            )

    def test_rejects_request_beyond_fixed_sylvester_fallback(self) -> None:
        left = _poly(
            ("x", "y"),
            (("1/1", (33, 0)), ("1/1", (0, 0))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (32, 0)), ("2/1", (0, 0))),
        )

        with pytest.raises(ValueError, match="Sylvester order"):
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )

    def test_bounds_abnormal_pseudo_remainder_coefficient_growth(self) -> None:
        def scaled_pair(exponent: int) -> tuple[RationalPolynomial, RationalPolynomial]:
            scale = 10**exponent
            left = _poly(
                ("x", "y"),
                (
                    (f"{scale}/1", (4, 0)),
                    (f"{scale}/1", (3, 0)),
                    (f"{scale}/1", (1, 0)),
                    (f"{3 * scale}/1", (0, 0)),
                ),
            )
            right = _poly(
                ("x", "y"),
                (
                    (f"{scale}/1", (3, 0)),
                    (f"{-2 * scale}/1", (1, 0)),
                    (f"{3 * scale}/1", (0, 0)),
                ),
            )
            return left, right

        boundary_left, boundary_right = scaled_pair(37)
        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=boundary_left,
                right=boundary_right,
                main_variable="x",
            )
        )
        assert tuple(member.degree_in_main_variable for member in result.members) == (
            4,
            3,
            2,
            1,
            0,
        )

        for exponent in (38, 250):
            beyond_left, beyond_right = scaled_pair(exponent)
            with pytest.raises(
                ValueError,
                match="Brown pseudo-remainder intermediate",
            ):
                MultivariateSubresultantSequenceRequest(
                    left=beyond_left,
                    right=beyond_right,
                    main_variable="x",
                )

    def test_rejects_unbounded_nonscalar_scaling_powers(self) -> None:
        """Abnormal-gap admission bounds the Brown kernel's scaling powers.

        The kernel raises nonscalar leading coefficients to the degree gaps
        (pseudo-remainder multipliers and abnormal scalar-subresultant
        powers), so their expanded support must be covered by the envelope
        before SymPy runs rather than being bounded only by the formal
        returned subresultant supports.
        """

        left = _poly(
            ("x", "y"),
            (("1/1", (6, 0)), ("-3/1", (4, 0)), ("1/1", (0, 0))),
        )
        right = _poly(
            ("x", "y"),
            (
                ("3/1", (2, 2)),
                ("2/1", (2, 1)),
                ("1/1", (2, 0)),
                ("1/1", (0, 0)),
            ),
        )

        with pytest.raises(ValueError, match="term-pair work budget"):
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )

    def test_admits_abnormal_drop_with_bounded_nonscalar_scaling(self) -> None:
        """A nonscalar scaling factor inside the derived power envelope runs."""

        left = _poly(
            ("x", "y"),
            (("1/1", (4, 0)), ("-3/1", (2, 0)), ("1/1", (0, 0))),
        )
        right = _poly(
            ("x", "y"),
            (
                ("2/1", (2, 2)),
                ("-3/1", (2, 1)),
                ("1/1", (2, 0)),
                ("1/1", (0, 0)),
            ),
        )

        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )

        assert result.source_order == "LEFT_RIGHT"
        assert tuple(member.degree_in_main_variable for member in result.members) == (
            4,
            2,
            0,
        )
        assert result.skipped_member_degrees == (1, 3)
        assert result.resultant == _poly(
            ("y",),
            (
                ("16/1", (8,)),
                ("-96/1", (7,)),
                ("296/1", (6,)),
                ("-576/1", (5,)),
                ("761/1", (4,)),
                ("-690/1", (3,)),
                ("415/1", (2,)),
                ("-150/1", (1,)),
                ("25/1", (0,)),
            ),
        )
        assert (
            MultivariateSubresultantSequenceResult.model_validate(
                result.model_dump(mode="json")
            )
            == result
        )

    def test_rejects_pseudo_remainder_chain_allowance_exceedance(self) -> None:
        """The scaling-power envelope carries the pseudo-remainder chain headroom.

        Every running pseudo-remainder coefficient keeps one source
        coefficient while appending one divisor-side factor per elimination
        step, so the folded power allowance adds one highest source
        remaining-variable degree.  Without that headroom term this request
        passes the support check and only the coarser term-pair proxy
        rejects it.
        """

        left = _poly(("x", "y"), (("1/1", (7, 0)), ("1/1", (0, 1))))
        right = _poly(("x", "y"), (("1/1", (6, 20)),))

        with pytest.raises(ValueError, match="intermediate polynomial-term budget"):
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )

    def test_rejects_unbounded_aggregate_sequence_support(self) -> None:
        variables = ("x", "y1", "y2", "y3", "y4", "y5", "y6", "y7")
        zeroes = (0,) * len(variables)
        left_terms = [("1/1", (2, *zeroes[1:]))]
        left_terms.extend(
            ("-1/1", (0, *(1 if index == offset else 0 for index in range(7))))
            for offset in range(7)
        )
        left = _poly(variables, tuple(left_terms))
        right = _poly(
            variables,
            (("1/1", (31, *zeroes[1:])), ("-1/1", zeroes)),
        )

        with pytest.raises(ValueError, match="subresultant sequence support"):
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )

    def test_result_parsing_is_structural_and_verification_rejects_forgery(
        self,
    ) -> None:
        left = _poly(
            ("x", "y"),
            (("1/1", (2, 0)), ("-1/1", (0, 1))),
        )
        right = _poly(
            ("x", "y"),
            (("1/1", (1, 0)), ("-1/1", (0, 1))),
        )
        result = compute_multivariate_subresultant_sequence(
            MultivariateSubresultantSequenceRequest(
                left=left,
                right=right,
                main_variable="x",
            )
        )
        original: dict[str, Any] = result.model_dump(mode="json")
        structural_errors: list[dict[str, Any]] = []
        forged_claims: list[dict[str, Any]] = []

        payload = deepcopy(original)
        payload["members"][-1]["polynomial"] = _scalar(
            ("x", "y"),
            "1",
        ).model_dump(mode="json")
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["members"][-1]["degree_in_main_variable"] = 1
        structural_errors.append(payload)

        payload = deepcopy(original)
        payload["left"]["polynomial"]["terms"][-1]["coefficient"]["num"] = "-2"
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["left"]["polynomial"]["terms"][0]["exponents"][0] = 100
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["source_order"] = "RIGHT_LEFT"
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["skipped_member_degrees"] = [1]
        structural_errors.append(payload)

        payload = deepcopy(original)
        payload["principal_subresultant_coefficients"][0]["coefficient"] = _scalar(
            ("y",), "1"
        ).model_dump(mode="json")
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["resultant"] = _scalar(("y",), "1").model_dump(mode="json")
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["resultant_sign_from_sequence_order"] = -1
        forged_claims.append(payload)

        payload = deepcopy(original)
        payload["gcd_member_index"] = 1
        structural_errors.append(payload)

        payload = deepcopy(original)
        payload["gcd_degree_in_main_variable"] = 1
        structural_errors.append(payload)

        payload = deepcopy(original)
        payload["gcd_member_leading_coefficient"] = _scalar(("y",), "1").model_dump(
            mode="json"
        )
        forged_claims.append(payload)

        assert _verify_multivariate_subresultant_sequence(result)
        for corrupted in structural_errors:
            with pytest.raises(ValueError):
                MultivariateSubresultantSequenceResult.model_validate(corrupted)
        for forged in forged_claims:
            parsed = MultivariateSubresultantSequenceResult.model_validate(forged)
            assert not _verify_multivariate_subresultant_sequence(parsed)
