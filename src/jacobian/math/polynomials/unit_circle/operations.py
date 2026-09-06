"""Exact bounded operations on rational polynomials on the unit circle."""

from __future__ import annotations

from fractions import Fraction
from math import lcm
from typing import Any

import sympy

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields import embeddings
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbeddingRecord,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
)
from jacobian.math.polynomials.unit_circle._models import (
    MAX_ARC_ENERGY_CONDUCTOR,
    MAX_ARC_ENERGY_DEGREE,
    MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS,
    MAX_ARC_ENERGY_FIELD_DEGREE,
    MAX_ARC_ENERGY_TERMS,
    UnitCircleArcEnergyRequest,
    UnitCircleArcEnergyResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    require_polynomial_budget,
)

__all__ = [
    "unit_circle_arc_energy",
    "verify_unit_circle_arc_energy",
]


def _fraction(value: CanonicalRational) -> Fraction:
    return value.as_fraction()


def _admit_polynomial(
    polynomial: RationalPolynomial,
) -> tuple[dict[int, Fraction], int]:
    if len(polynomial.variables) != 1:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.unit_circle.univariate_required",
            message="unit-circle operations require one polynomial variable",
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_ARC_ENERGY_TERMS,
        maximum_exponent=MAX_ARC_ENERGY_DEGREE,
        maximum_coefficient_digits=MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS,
        label="unit-circle polynomial",
    )
    coefficients = {
        term.exponents[0]: _fraction(term.coefficient)
        for term in polynomial.polynomial.terms
    }
    return coefficients, max(coefficients, default=0)


def _admit_arc(
    request: UnitCircleArcEnergyRequest,
) -> tuple[dict[int, Fraction], int, Fraction, int]:
    coefficients, degree = _admit_polynomial(request.polynomial)
    start = _fraction(request.start_turn)
    end = _fraction(request.end_turn)
    width = end - start
    if width < 0 or width > 1:
        raise OperationDomainValidationError(
            location=("end_turn",),
            code="polynomial.unit_circle.arc_width",
            message="oriented arc width must satisfy 0 <= end_turn-start_turn <= 1",
        )
    denominator = lcm(int(request.start_turn.den), int(request.end_turn.den))
    if denominator > MAX_ARC_ENERGY_CONDUCTOR:
        raise OperationDomainValidationError(
            location=("start_turn", "end_turn"),
            code="polynomial.unit_circle.conductor_bound",
            message="rational turn denominators exceed the cyclotomic conductor bound",
        )
    return coefficients, degree, width, denominator


def _minimal_polynomial(expr: Any) -> tuple[str, ...]:
    variable = sympy.Symbol("x")
    polynomial = sympy.Poly(
        sympy.minimal_polynomial(expr, variable), variable, domain=sympy.QQ
    )
    _denominator, integral = polynomial.clear_denoms(convert=True)
    _content, primitive = integral.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    coefficients = tuple(str(int(value)) for value in primitive.all_coeffs())
    if len(coefficients) - 1 > MAX_ARC_ENERGY_FIELD_DEGREE:
        raise OperationDomainValidationError(
            location=("pi_inverse_coefficient",),
            code="polynomial.unit_circle.field_degree_bound",
            message="the real cyclotomic coefficient exceeds the field-degree bound",
        )
    if any(
        len(value.lstrip("-")) > MAX_ARC_ENERGY_FIELD_COEFFICIENT_DIGITS
        for value in coefficients
    ):
        raise OperationDomainValidationError(
            location=("pi_inverse_coefficient",),
            code="polynomial.unit_circle.field_height_bound",
            message="the real cyclotomic coefficient field exceeds the height bound",
        )
    return coefficients


def _embedding_record(
    presentation: SimpleNumberFieldPresentation, expr: Any
) -> RealNumberFieldEmbeddingRecord:
    profile = embeddings(presentation)
    approximation = float(sympy.N(expr, 30)) if expr != 0 else 0.0
    for candidate in profile.records:
        if not isinstance(candidate, RealNumberFieldEmbeddingRecord):
            continue
        lower = float(candidate.isolating_interval.lower.as_fraction())
        upper = float(candidate.isolating_interval.upper.as_fraction())
        if lower < approximation < upper:
            return candidate
    real_records = tuple(
        candidate
        for candidate in profile.records
        if isinstance(candidate, RealNumberFieldEmbeddingRecord)
    )
    if not real_records:
        raise OperationDomainValidationError(
            location=("coefficient",),
            code="polynomial.unit_circle.real_embedding",
            message="the exact coefficient has no real embedding",
        )
    return min(
        real_records,
        key=lambda candidate: abs(
            float(candidate.isolating_interval.lower.as_fraction()) - approximation
        ),
    )


def _binding_in_field(
    expr: Any,
    alpha: Any,
    presentation: SimpleNumberFieldPresentation,
    record: RealNumberFieldEmbeddingRecord,
) -> SimpleNumberFieldRealEmbeddingBinding:
    if expr == 0:
        element_coordinates = (
            CanonicalRational(num="0", den="1"),
        ) * presentation.degree
    else:
        algebraic = sympy.to_number_field(expr, alpha)
        descending = tuple(algebraic.coeffs())
        ascending = tuple(reversed(descending)) + (sympy.Rational(0),) * (
            presentation.degree - len(descending)
        )
        element_coordinates = tuple(
            CanonicalRational.from_fraction(Fraction(int(value.p), int(value.q)))
            for value in ascending
        )
    return SimpleNumberFieldRealEmbeddingBinding(
        element=SimpleNumberFieldElement(
            presentation=presentation,
            coefficients_ascending=element_coordinates,
        ),
        embedding_record=record,
    )


def _binding(expr: Any) -> SimpleNumberFieldRealEmbeddingBinding:
    coefficients = ("1", "0") if expr == 0 else _minimal_polynomial(expr)
    presentation = SimpleNumberFieldPresentation(coefficients_descending=coefficients)
    record = _embedding_record(presentation, expr)
    return _binding_in_field(expr, expr, presentation, record)


def _correlations(
    coefficients: dict[int, Fraction], degree: int
) -> dict[int, Fraction]:
    return {
        shift: sum(
            (
                coefficients.get(index + shift, Fraction(0))
                * coefficients.get(index, Fraction(0))
                for index in range(degree - shift + 1)
            ),
            Fraction(0),
        )
        for shift in range(degree + 1)
    }


def unit_circle_arc_energy(
    request: UnitCircleArcEnergyRequest,
) -> UnitCircleArcEnergyResult:
    """Return ``integral |P(exp(2*pi*i*t))|^2 dt`` on an oriented arc."""
    coefficients, degree, width, _conductor = _admit_arc(request)
    correlations = _correlations(coefficients, degree)
    rational_part = CanonicalRational.from_fraction(width * correlations[0])
    pi_coefficient = sum(
        sympy.Rational(value.numerator, value.denominator)
        / shift
        * (
            sympy.sin(
                2
                * sympy.pi
                * shift
                * sympy.Rational(request.end_turn.num, request.end_turn.den)
            )
            - sympy.sin(
                2
                * sympy.pi
                * shift
                * sympy.Rational(request.start_turn.num, request.start_turn.den)
            )
        )
        for shift, value in correlations.items()
        if shift > 0 and value
    )
    return UnitCircleArcEnergyResult(
        polynomial=request.polynomial,
        start_turn=request.start_turn,
        end_turn=request.end_turn,
        rational_part=rational_part,
        pi_inverse_coefficient=_binding(sympy.simplify(pi_coefficient)),
    )


def verify_unit_circle_arc_energy(claim: UnitCircleArcEnergyResult) -> bool:
    """Verify the exact arc-energy identity against its retained source."""
    try:
        return (
            unit_circle_arc_energy(
                UnitCircleArcEnergyRequest(
                    polynomial=claim.polynomial,
                    start_turn=claim.start_turn,
                    end_turn=claim.end_turn,
                )
            )
            == claim
        )
    except (OperationDomainValidationError, ValueError, RuntimeError):
        return False
