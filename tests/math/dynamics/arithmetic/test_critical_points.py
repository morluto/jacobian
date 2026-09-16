"""Tests for exact polynomial critical-point profiles."""

from __future__ import annotations

import json
from copy import deepcopy
from fractions import Fraction

import pytest
import sympy
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.dynamics.arithmetic import (
    critical_points,
    polynomial_coefficients,
    polynomial_from_coefficients,
)
from jacobian.math.dynamics.arithmetic._models import (
    MAX_CRITICAL_DEGREE,
    CriticalPointsRequest,
    CriticalPointsResult,
)
from jacobian.math.dynamics.arithmetic._tools import (
    compute_critical_points,
    verify_critical_points,
)
from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy


def _p(*coefficients: int) -> object:
    return polynomial_from_coefficients(coefficients)


def _profile(result: CriticalPointsResult) -> tuple[tuple[int, int], ...]:
    return tuple((factor.degree, factor.multiplicity) for factor in result.factors)


def _conjugate(coefficients: tuple[int, ...], a: int, b: int):
    x = sympy.Symbol("x")
    expression = sum(
        coefficient * x**index for index, coefficient in enumerate(coefficients)
    )
    conjugated = sympy.expand((expression.subs(x, a * x + b) - b) / a)
    polynomial = sympy.Poly(conjugated, x, domain=sympy.QQ)
    rationals = [sympy.Rational(value) for value in polynomial.all_coeffs()]
    return polynomial_from_coefficients(
        tuple(Fraction(int(value.p), int(value.q)) for value in reversed(rationals))
    )


class TestKnownAnswers:
    def test_quadratic_has_single_simple_critical_point(self) -> None:
        result = compute_critical_points(CriticalPointsRequest(polynomial=_p(0, 0, 1)))

        assert (result.degree, result.derivative_degree) == (2, 1)
        assert _profile(result) == ((1, 1),)
        assert result.distinct_finite_critical_points == 1
        assert result.total_multiplicity == 1
        assert result.infinity_is_critical is True
        assert result.affine is False

    def test_cubic_has_two_simple_critical_points(self) -> None:
        result = compute_critical_points(
            CriticalPointsRequest(polynomial=_p(0, -3, 0, 1))
        )

        assert result.total_multiplicity == 2
        assert _profile(result) == ((1, 1), (1, 1))
        coefficients = [
            polynomial_coefficients(factor.factor) for factor in result.factors
        ]
        assert sorted(coefficients) == [
            (Fraction(-1), Fraction(1)),
            (Fraction(1), Fraction(1)),
        ]

    def test_power_map_has_one_critical_point_of_full_multiplicity(self) -> None:
        result = compute_critical_points(
            CriticalPointsRequest(polynomial=_p(0, 0, 0, 0, 1))
        )

        assert result.degree == 4
        assert _profile(result) == ((1, 3),)
        assert result.distinct_finite_critical_points == 1
        assert result.total_multiplicity == 3

    def test_affine_map_has_no_finite_critical_points(self) -> None:
        result = compute_critical_points(CriticalPointsRequest(polynomial=_p(1, 2)))

        assert result.degree == 1
        assert result.affine is True
        assert result.infinity_is_critical is False
        assert result.factors == ()
        assert result.distinct_finite_critical_points == 0
        assert result.total_multiplicity == 0

    def test_critical_multiplier_is_exactly_zero(self) -> None:
        result = compute_critical_points(CriticalPointsRequest(polynomial=_p(0, 0, 1)))

        assert result.finite_critical_multiplier.as_fraction() == 0


class TestDefiningInvariants:
    @pytest.mark.parametrize("degree", [1, 2, 3, 5, 8])
    def test_total_multiplicity_equals_degree_minus_one(self, degree: int) -> None:
        coefficients = (0,) * degree + (1,)
        result = compute_critical_points(
            CriticalPointsRequest(polynomial=_p(*coefficients))
        )

        assert result.degree == degree
        assert result.derivative_degree == degree - 1
        assert result.total_multiplicity == degree - 1
        assert result.infinity_is_critical == (degree >= 2)

    def test_factor_reconstruction_matches_the_derivative(self) -> None:
        source = polynomial_from_coefficients((0, -3, 0, 1))
        result = critical_points(source)
        x = sympy.Symbol("x")
        reconstructed = sympy.Integer(1)
        for factor in result.factors:
            expression = sum(
                Fraction(term.coefficient.as_fraction()) * x ** term.exponents[0]
                for term in factor.factor.polynomial.terms
            )
            reconstructed *= expression**factor.multiplicity

        derivative = sum(
            Fraction(term.coefficient.as_fraction()) * x ** term.exponents[0]
            for term in result.derivative.polynomial.terms
        )
        # Monic reconstruction equals the monic derivative up to the leading
        # coefficient of the derivative.
        lead = Fraction(polynomial_coefficients(result.derivative)[-1])
        assert sympy.expand(reconstructed - derivative / lead) == 0

    def test_affine_conjugation_transports_the_profile(self) -> None:
        coefficients = (1, 0, 1)
        source = polynomial_from_coefficients(coefficients)
        conjugated = _conjugate(coefficients, 2, 3)

        base = critical_points(source)
        transported = critical_points(conjugated)

        assert _profile(base) == _profile(transported)
        assert base.total_multiplicity == transported.total_multiplicity
        assert base.affine == transported.affine


class TestAdmissionAndParity:
    def test_native_and_catalog_paths_agree(self) -> None:
        source = polynomial_from_coefficients((0, -3, 0, 1))

        assert compute_critical_points(
            CriticalPointsRequest(polynomial=source)
        ) == critical_points(source)

    def test_round_trip_and_forgery(self) -> None:
        source = polynomial_from_coefficients((0, -3, 0, 1))
        result = compute_critical_points(CriticalPointsRequest(polynomial=source))
        restored = CriticalPointsResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert verify_critical_points(restored)
        forged = deepcopy(restored.model_dump(mode="json"))
        forged["total_multiplicity"] = 0
        with pytest.raises(ValidationError):
            CriticalPointsResult.model_validate_json(json.dumps(forged))
        forged_factors = deepcopy(restored.model_dump(mode="json"))
        forged_factors["distinct_finite_critical_points"] = 5
        with pytest.raises(ValidationError):
            CriticalPointsResult.model_validate_json(json.dumps(forged_factors))

    def test_degree_bound_is_a_resource_boundary(self) -> None:
        oversized = (0,) * (MAX_CRITICAL_DEGREE + 1) + (1,)
        source = polynomial_from_coefficients(oversized)

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            compute_critical_points(CriticalPointsRequest(polynomial=source))
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.critical_points.degree_bound"
        )

    def test_constant_polynomial_is_a_domain_boundary(self) -> None:
        source = polynomial_from_coefficients((5,))

        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_critical_points(CriticalPointsRequest(polynomial=source))
        assert (
            exc_info.value.errors()[0]["type"]
            == "arithmetic_dynamics.critical_points.nonconstant_required"
        )

    def test_non_univariate_polynomial_is_rejected(self) -> None:
        x, y = sympy.symbols("x y")
        bivariate = rational_polynomial_from_sympy(
            sympy.Poly(x + y, x, y, domain=sympy.QQ), ("x", "y")
        )

        with pytest.raises(ValueError, match="univariate"):
            critical_points(bivariate)
