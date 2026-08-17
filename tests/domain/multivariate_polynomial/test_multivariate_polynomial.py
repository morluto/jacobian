"""Tests for bounded exact multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

import pytest

from jacobian.contracts.multivariate_polynomial import (
    MultivariateDivisionRequest,
    MultivariateGcdRequest,
    MultivariateResultantRequest,
)
from jacobian.domains.multivariate_polynomial.operations import (
    compute_multivariate_division,
    compute_multivariate_gcd,
    compute_multivariate_resultant,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _poly(
    variables: tuple[str, ...],
    terms: tuple[tuple[str, tuple[int, ...]], ...],
) -> RationalPolynomial:
    """Build a ``RationalPolynomial`` from ``"num/den", (exponents,)`` tuples."""

    return RationalPolynomial.model_validate(
        {
            "polynomial_schema_version": "1",
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
        import pytest
        from pydantic import ValidationError

        left = _poly(("x", "y"), (("1/1", (1, 0)),))
        right = _poly(("x", "y"), ())
        with pytest.raises(ValidationError, match="nonzero"):
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

    def test_resultant_rejects_zero_degree(self) -> None:
        """The elimination variable must appear in both polynomials."""

        # left has zero degree in x; right has zero degree in x
        left = _poly(("x", "y"), (("1/1", (0, 1)),))
        right = _poly(("x", "y"), (("1/1", (0, 1)),))
        with pytest.raises(ValueError, match="zero degree in the elimination variable"):
            MultivariateResultantRequest(
                left=left, right=right, elimination_variable="x"
            )

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
