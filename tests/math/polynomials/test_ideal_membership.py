"""Contract and mathematical tests for ideal membership and normal form."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._models import IdealMembershipRequest
from jacobian.math.polynomials._operations import (
    polynomial_ideal_membership,
    polynomial_ideal_normal_form,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


_XY_VARS = ("x", "y")


def _ideal1() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=("x",),
        generators=(_poly(("x",), {(2,): 1}),),
    )


def _ideal_xy() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=_XY_VARS,
        generators=(
            _poly(_XY_VARS, {(1, 0): 1, (0, 1): 1}),
        ),
    )


def _ideal_line_through_origin() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=_XY_VARS,
        generators=(
            _poly(_XY_VARS, {(1, 0): 1}),
            _poly(_XY_VARS, {(0, 1): 1, (1, 0): -1}),
        ),
    )


def _ideal_coordinate_axes() -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(
        variables=_XY_VARS,
        generators=(
            _poly(_XY_VARS, {(1, 0): 1}),
            _poly(_XY_VARS, {(0, 1): 1}),
        ),
    )


def test_operations_in_catalog() -> None:
    ids = {tool.operation_id for tool in POLYNOMIAL_INVARIANT_OPERATIONS}
    assert "polynomial.ideal.membership.decide" in ids
    assert "polynomial.ideal.normal_form.compute" in ids


def test_member_is_in_ideal() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(2,): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.in_ideal is True
    assert len(result.normal_form.polynomial.terms) == 0


def test_non_member_is_not_in_ideal() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(1,): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.in_ideal is False
    assert len(result.normal_form.polynomial.terms) == 1


def test_zero_is_in_any_ideal() -> None:
    req = IdealMembershipRequest(
        ideal=_ideal_xy(),
        polynomial=_poly(("x", "y"), {}),
    )
    result = polynomial_ideal_membership(req)
    assert result.in_ideal is True


def test_normal_form_returns_canonical_remainder() -> None:
    # x + 1 mod <x^2> = x + 1
    req = IdealMembershipRequest(
        ideal=_ideal1(),
        polynomial=_poly(("x",), {(1,): 1, (0,): 1}),
    )
    result = polynomial_ideal_normal_form(req)
    assert result.monomial_order == "grevlex"
    assert len(result.remainder.polynomial.terms) == 2
    assert result.remainder.polynomial.terms[0].exponents == (1,)


def test_membership_rejects_mismatched_rings() -> None:
    with pytest.raises(ValueError, match="same ordered ring"):
        IdealMembershipRequest(
            ideal=_ideal1(),
            polynomial=_poly(("y",), {(1,): 1}),
        )


def test_multivariate_membership() -> None:
    # Is y in <x, y-x> = <x, y>?
    req = IdealMembershipRequest(
        ideal=_ideal_line_through_origin(),
        polynomial=_poly(("x", "y"), {(0, 1): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.in_ideal is True


def test_multivariate_non_membership() -> None:
    # Is 1 in <x, y>? No.
    req = IdealMembershipRequest(
        ideal=_ideal_coordinate_axes(),
        polynomial=_poly(("x", "y"), {(0, 0): 1}),
    )
    result = polynomial_ideal_membership(req)
    assert result.in_ideal is False
