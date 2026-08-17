from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomial_interpolation import (
    MultipointEvaluationRequest,
    NewtonInterpolationRequest,
    RationalPoint,
)
from jacobian.domains.polynomial_interpolation.operations import (
    compute_multipoint_evaluate,
    compute_newton_interpolation,
)

R = CanonicalRational


def _pt(x: tuple[str, str], y: tuple[str, str]) -> RationalPoint:
    return RationalPoint(x=R(num=x[0], den=x[1]), y=R(num=y[0], den=y[1]))


def test_newton_two_points_linear() -> None:
    """Interpolate (0,1) and (1,2): polynomial x + 1."""
    result = compute_newton_interpolation(
        NewtonInterpolationRequest(
            points=[_pt(("0", "1"), ("1", "1")), _pt(("1", "1"), ("2", "1"))]
        )
    )
    coeffs = [c.as_fraction() for c in result.coefficients]
    assert coeffs == [Fraction(1), Fraction(1)]


def test_newton_three_points_quadratic() -> None:
    """Interpolate (0,0), (1,1), (2,4): polynomial x^2."""
    result = compute_newton_interpolation(
        NewtonInterpolationRequest(
            points=[
                _pt(("0", "1"), ("0", "1")),
                _pt(("1", "1"), ("1", "1")),
                _pt(("2", "1"), ("4", "1")),
            ]
        )
    )
    coeffs = [c.as_fraction() for c in result.coefficients]
    assert coeffs == [Fraction(0), Fraction(0), Fraction(1)]


def test_newton_constant() -> None:
    """Interpolate a single point: constant polynomial."""
    result = compute_newton_interpolation(
        NewtonInterpolationRequest(points=(_pt(("5", "1"), ("3", "1")),))
    )
    coeffs = [c.as_fraction() for c in result.coefficients]
    assert coeffs == [Fraction(3)]


def test_newton_returns_divided_differences() -> None:
    """The divided differences are returned alongside the coefficients."""
    result = compute_newton_interpolation(
        NewtonInterpolationRequest(
            points=[
                _pt(("0", "1"), ("1", "1")),
                _pt(("1", "1"), ("2", "1")),
                _pt(("2", "1"), ("5", "1")),
            ]
        )
    )
    assert len(result.divided_differences) == 3
    assert result.method == "NEWTON_DIVIDED_DIFFERENCE"


def test_newton_reproduces_values() -> None:
    """The interpolating polynomial passes through all data points."""
    points = [
        _pt(("0", "1"), ("1", "1")),
        _pt(("1", "1"), ("2", "1")),
        _pt(("2", "1"), ("5", "1")),
        _pt(("3", "1"), ("10", "1")),
    ]
    result = compute_newton_interpolation(
        NewtonInterpolationRequest(points=tuple(points))
    )
    coeffs = [c.as_fraction() for c in result.coefficients]
    for x, y in [(0, 1), (1, 2), (2, 5), (3, 10)]:
        val = sum(c * Fraction(x) ** i for i, c in enumerate(coeffs))
        assert val == Fraction(y)


def test_multipoint_evaluate_quadratic() -> None:
    """Evaluate x^2 + 1 at x=0,1,2: [1, 2, 5]."""
    result = compute_multipoint_evaluate(
        MultipointEvaluationRequest(
            coefficients=(
                R(num="1", den="1"),
                R(num="0", den="1"),
                R(num="1", den="1"),
            ),
            evaluation_points=(
                R(num="0", den="1"),
                R(num="1", den="1"),
                R(num="2", den="1"),
            ),
        )
    )
    values = [v.as_fraction() for v in result.values]
    assert values == [Fraction(1), Fraction(2), Fraction(5)]


def test_multipoint_evaluate_constant() -> None:
    """Evaluate constant polynomial 5 at any point: [5]."""
    result = compute_multipoint_evaluate(
        MultipointEvaluationRequest(
            coefficients=(R(num="5", den="1"),),
            evaluation_points=(R(num="3", den="1"),),
        )
    )
    assert result.values[0].as_fraction() == Fraction(5)


def test_multipoint_evaluate_single_point() -> None:
    """Evaluate at a single point."""
    result = compute_multipoint_evaluate(
        MultipointEvaluationRequest(
            coefficients=(
                R(num="1", den="1"),
                R(num="2", den="1"),
                R(num="3", den="1"),
            ),
            evaluation_points=(R(num="1", den="1"),),
        )
    )
    # 3*1^2 + 2*1 + 1 = 6
    assert result.values[0].as_fraction() == Fraction(6)


def test_contract_rejects_duplicate_x() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        NewtonInterpolationRequest(
            points=(
                _pt(("1", "1"), ("2", "1")),
                _pt(("1", "1"), ("3", "1")),
            )
        )
