"""Tests for Groebner basis, normal form, and elimination ideal operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.commutative_algebra_ops import _operations
from jacobian.math.commutative_algebra_ops._models import (
    EliminationIdealRequest,
    EliminationIdealResult,
    GroebnerBasisRequest,
    IdealNormalFormRequest,
    IdealNormalFormResult,
)
from jacobian.math.commutative_algebra_ops._operations import (
    compute_elimination_ideal,
    compute_groebner_basis,
    compute_ideal_normal_form,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
)


def _poly(
    variables: tuple[str, ...],
    *terms: tuple[int, int, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": str(num), "den": str(den)},
                        "exponents": list(exp),
                    }
                    for num, den, exp in terms
                ]
            },
        }
    )


def _ideal(
    variables: tuple[str, ...],
    generators: tuple[RationalPolynomial, ...],
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(variables=variables, generators=generators)


class TestGroebnerBasis:
    """Tests for ``polynomial.ideal.groebner_basis.compute``."""

    def test_simple_ideal(self):
        """Gröbner basis of <x^2 - y, xy - 1> has a finite basis."""
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 1)), (-1, 1, (0, 0)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = compute_groebner_basis(
            GroebnerBasisRequest(ideal=ideal, monomial_order="grevlex")
        )
        assert result.generator_count >= 1
        assert result.generator_count == len(result.basis.generators)

    def test_principal_ideal(self):
        """Gröbner basis of <x> in Q[x] is <x>."""
        g = _poly(("x",), (1, 1, (1,)))
        ideal = _ideal(("x",), (g,))
        result = compute_groebner_basis(
            GroebnerBasisRequest(ideal=ideal, monomial_order="lex")
        )
        assert result.generator_count >= 1

    def test_lex_order(self):
        """Gröbner basis with lex order works."""
        g1 = _poly(("x", "y"), (1, 1, (1, 1)))
        g2 = _poly(("x", "y"), (1, 1, (1, 0)), (-1, 1, (0, 1)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = compute_groebner_basis(
            GroebnerBasisRequest(ideal=ideal, monomial_order="lex")
        )
        assert result.generator_count >= 1


class TestIdealNormalForm:
    """Tests for ``polynomial.ideal.normal_form.compute``."""

    def test_polynomial_in_ideal(self):
        """x^2 mod <x^2 - y^2> should give a nonzero remainder that is not in the ideal."""
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        ideal = _ideal(("x", "y"), (g,))
        poly = _poly(("x", "y"), (1, 1, (2, 0)))
        result = compute_ideal_normal_form(
            IdealNormalFormRequest(ideal=ideal, polynomial=poly)
        )
        assert result.in_ideal is False
        assert len(result.remainder.polynomial.terms) > 0

    def test_polynomial_in_ideal_exactly(self):
        """x^2 - y^2 mod <x^2 - y^2> should give zero (in the ideal)."""
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        ideal = _ideal(("x", "y"), (g,))
        poly = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = compute_ideal_normal_form(
            IdealNormalFormRequest(ideal=ideal, polynomial=poly)
        )
        assert result.in_ideal is True
        assert len(result.remainder.polynomial.terms) == 0

    def test_constant_in_unit_ideal(self):
        """A constant is in the ideal <1> = Q[x,y]."""
        g = _poly(("x", "y"), (1, 1, (0, 0)))
        ideal = _ideal(("x", "y"), (g,))
        poly = _poly(("x", "y"), (3, 1, (0, 0)))
        result = compute_ideal_normal_form(
            IdealNormalFormRequest(ideal=ideal, polynomial=poly)
        )
        assert result.in_ideal is True


class TestEliminationIdeal:
    """Tests for ``polynomial.ideal.elimination.compute``."""

    def test_eliminate_one_variable(self):
        """Eliminate x from <x^2 - y^2, x + y> → get ideal in Q[y]."""
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        g2 = _poly(("x", "y"), (1, 1, (1, 0)), (1, 1, (0, 1)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = compute_elimination_ideal(
            EliminationIdealRequest(ideal=ideal, eliminated_variables=("x",))
        )
        assert "x" not in result.elimination_ideal.variables
        assert len(result.elimination_ideal.generators) >= 1

    def test_eliminated_variables_not_in_result(self):
        """The elimination ideal should not contain eliminated variables."""
        g1 = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        g2 = _poly(("x", "y"), (1, 1, (1, 0)), (1, 1, (0, 1)))
        ideal = _ideal(("x", "y"), (g1, g2))
        result = compute_elimination_ideal(
            EliminationIdealRequest(ideal=ideal, eliminated_variables=("x",))
        )
        for var in result.elimination_ideal.variables:
            assert var != "x"


class TestTypedTimeoutOutcomes:
    """Budget expiry must surface as a typed outcome, not a host exception."""

    def test_normal_form_timeout_returns_typed_outcome(self, monkeypatch):
        """An expired normal-form budget returns TIMEOUT instead of raising."""

        def exceed_budget(*args, **kwargs):
            raise _operations._NormalFormTimeoutError("budget expired")

        monkeypatch.setattr(_operations.sympy, "groebner", exceed_budget)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = compute_ideal_normal_form(
            IdealNormalFormRequest(
                ideal=_ideal(("x", "y"), (g,)),
                polynomial=_poly(("x", "y"), (1, 1, (2, 0))),
            )
        )
        assert result.outcome == "TIMEOUT"
        assert result.remainder is None
        assert result.detail is not None

    def test_elimination_timeout_returns_typed_outcome(self, monkeypatch):
        """An expired elimination budget returns TIMEOUT instead of raising."""

        def exceed_budget(*args, **kwargs):
            raise _operations._EliminationTimeoutError("budget expired")

        monkeypatch.setattr(_operations.sympy, "groebner", exceed_budget)
        g = _poly(("x", "y"), (1, 1, (2, 0)), (-1, 1, (0, 2)))
        result = compute_elimination_ideal(
            EliminationIdealRequest(
                ideal=_ideal(("x", "y"), (g,)),
                eliminated_variables=("x",),
            )
        )
        assert result.outcome == "TIMEOUT"
        assert result.elimination_ideal is None
        assert result.eliminated_variables == ("x",)
        assert result.detail is not None

    def test_timed_out_normal_form_cannot_carry_a_remainder(self):
        remainder = _poly(("x", "y"), (1, 1, (2, 0)))
        with pytest.raises(ValidationError, match="timed-out"):
            IdealNormalFormResult(outcome="TIMEOUT", remainder=remainder)

    def test_computed_normal_form_requires_a_remainder(self):
        with pytest.raises(ValidationError, match="requires a remainder"):
            IdealNormalFormResult()

    def test_timed_out_elimination_cannot_carry_an_ideal(self):
        elimination_ideal = _ideal(
            ("y",),
            (_poly(("y",), (1, 1, (2,))),),
        )
        with pytest.raises(ValidationError, match="timed-out"):
            EliminationIdealResult(
                outcome="TIMEOUT",
                elimination_ideal=elimination_ideal,
                eliminated_variables=("x",),
                detail="budget expired",
            )

    def test_computed_elimination_requires_an_ideal(self):
        with pytest.raises(ValidationError, match="requires an ideal"):
            EliminationIdealResult(eliminated_variables=("x",))
