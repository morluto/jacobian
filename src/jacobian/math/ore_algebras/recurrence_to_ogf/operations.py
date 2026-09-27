"""Bounded exact ordinary-generating-function recurrence transform."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import comb, gcd
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.sequences.core._models import FiniteRationalSequence
from jacobian.math.ore_algebras._models import (
    MAX_DIFFERENTIAL_ORDER,
    MAX_SHIFT_COEFFICIENT_DEGREE,
    MAX_SHIFT_COEFFICIENT_DIGITS,
    MAX_SHIFT_COEFFICIENT_TERMS,
    DifferentialOreOperator,
    ShiftOreOperator,
)
from jacobian.math.ore_algebras.recurrence_to_ogf._models import (
    RecurrenceOGFEquation,
    RecurrenceOGFEquationRequest,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_canonical_rational_function,
)

_Poly = dict[int, Fraction]
MAX_RECURRENCE_OGF_WORK_CELLS = 100_000


def _decode_polynomial(value: SparseRationalPolynomial) -> _Poly:
    return {
        int(term.exponents[0]): term.coefficient.as_fraction() for term in value.terms
    }


def _encode_polynomial(value: _Poly) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=CanonicalRational.from_fraction(coefficient),
                exponents=(exponent,),
            )
            for exponent, coefficient in sorted(value.items(), reverse=True)
            if coefficient
        )
    )


def _encode_rational_polynomial(value: _Poly, variable: str) -> RationalFunction:
    return RationalFunction(
        domain="QQ",
        variables=(variable,),
        numerator=_encode_polynomial(value),
        denominator=_encode_polynomial({0: Fraction(1)}),
    )


def _shift_polynomial(value: _Poly, step: int) -> _Poly:
    """Expand p(z+step) exactly in the monomial basis."""
    shifted: _Poly = {}
    for exponent, coefficient in value.items():
        for target in range(exponent + 1):
            contribution = (
                coefficient * comb(exponent, target) * step ** (exponent - target)
            )
            shifted[target] = shifted.get(target, Fraction(0)) + contribution
            if shifted[target] == 0:
                del shifted[target]
    return shifted


def _evaluate_polynomial(value: _Poly, point: int) -> Fraction:
    return sum(
        (coefficient * point**degree for degree, coefficient in value.items()),
        Fraction(0),
    )


def _digits_for_bit_bound(bits: int) -> int:
    # 0.30103 is a strict rational upper approximation for log10(2).
    return (bits * 30_103) // 100_000 + 1


def _weighted_coefficient_height_bits(
    contributions: list[tuple[Fraction, int]],
) -> tuple[int, int]:
    """Bound one transformed coefficient after its exact integer weights."""
    denominator = 1
    for value, _weight in contributions:
        factor = value.denominator
        denominator = denominator // gcd(denominator, factor) * factor
    numerator_terms = [
        abs(value.numerator * weight * (denominator // value.denominator)).bit_length()
        for value, weight in contributions
    ]
    numerator_bits = max(numerator_terms, default=1) + max(
        0, len(contributions) - 1
    ).bit_length()
    return numerator_bits, denominator.bit_length()


def _transformed_coefficient_bounds(
    polynomials: list[tuple[int, _Poly]], shift_order: int
) -> tuple[tuple[int, int], ...]:
    maximum_degree = max((max(polynomial, default=0) for _, polynomial in polynomials), default=0)
    stirling: list[list[int]] = [[1]]
    for power in range(1, maximum_degree + 1):
        previous = stirling[-1]
        row = [0] * (power + 1)
        for degree in range(1, power + 1):
            same_degree = previous[degree] if degree < len(previous) else 0
            row[degree] = degree * same_degree + previous[degree - 1]
        stirling.append(row)
    slots: dict[tuple[int, int], list[tuple[Fraction, int]]] = {}
    for shift, polynomial in polynomials:
        for source_degree, coefficient in polynomial.items():
            for theta_degree in range(source_degree + 1):
                shift_factor = comb(source_degree, theta_degree) * (-shift) ** (
                    source_degree - theta_degree
                )
                if not shift_factor:
                    continue
                for derivative_order in range(theta_degree + 1):
                    multiplier = shift_factor * stirling[theta_degree][derivative_order]
                    if multiplier:
                        slot = (
                            derivative_order,
                            shift_order - shift + derivative_order,
                        )
                        slots.setdefault(slot, []).append((coefficient, multiplier))
    return tuple(
        _weighted_coefficient_height_bits(contributions)
        for contributions in slots.values()
    )


def _product_numerator_bits(left: Fraction, right: Fraction) -> int:
    if abs(left.numerator) == 1:
        return abs(right.numerator).bit_length()
    if abs(right.numerator) == 1:
        return abs(left.numerator).bit_length()
    return abs(left.numerator).bit_length() + abs(right.numerator).bit_length()


def _canonical_request(
    recurrence: ShiftOreOperator | Mapping[str, Any],
    initial_coefficients: FiniteRationalSequence | Mapping[str, Any],
) -> tuple[RecurrenceOGFEquationRequest, ShiftOreOperator]:
    try:
        request = RecurrenceOGFEquationRequest.model_validate(
            {
                "recurrence": recurrence.model_dump()
                if isinstance(recurrence, ShiftOreOperator)
                else recurrence,
                "initial_coefficients": initial_coefficients.model_dump()
                if isinstance(initial_coefficients, FiniteRationalSequence)
                else initial_coefficients,
            }
        )
        operator = ShiftOreOperator.model_validate(request.recurrence.model_dump())
        for index, term in enumerate(operator.terms):
            require_canonical_rational_function(
                term.coefficient,
                maximum_terms=MAX_SHIFT_COEFFICIENT_TERMS,
                maximum_exponent=MAX_SHIFT_COEFFICIENT_DEGREE,
                maximum_coefficient_digits=MAX_SHIFT_COEFFICIENT_DIGITS,
                label=f"recurrence coefficient {index}",
            )
        return request, operator
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.recurrence_ogf_request",
            message=(
                "the recurrence must be nonzero with canonical polynomial "
                "coefficients in QQ[n] and exactly one initial coefficient per "
                "shift below its largest exponent"
            ),
        ) from exc


def _admit_transform(
    request: RecurrenceOGFEquationRequest,
    operator: ShiftOreOperator,
) -> tuple[list[tuple[int, _Poly]], int, _Poly]:
    """Bound output shape, coefficient height, work, and bytes before expansion."""
    polynomials: list[tuple[int, _Poly]] = []
    maximum_degree = 0
    maximum_output_degree = 0
    work = (operator.order + 1) * (len(operator.terms) + 1)
    for term in operator.terms:
        coefficient = term.coefficient
        numerator = _decode_polynomial(coefficient.numerator)
        denominator = _decode_polynomial(coefficient.denominator)
        if denominator != {0: Fraction(1)}:
            raise OperationDomainValidationError(
                location=("recurrence", "terms", term.exponent, "coefficient"),
                code="ore_algebra.recurrence_ogf_polynomial_domain",
                message="ordinary generating-function conversion requires coefficients in QQ[n]",
            )
        degree = max(numerator, default=0)
        maximum_degree = max(maximum_degree, degree)
        maximum_output_degree = max(
            maximum_output_degree, operator.order - term.exponent + degree
        )
        polynomials.append((term.exponent, numerator))
        work += len(numerator) * (degree + 1) ** 2
    work += sum(shift * len(polynomial) for shift, polynomial in polynomials)
    if maximum_degree > MAX_DIFFERENTIAL_ORDER:
        raise OperationResourceAdmissionError(
            location=("recurrence", "terms"),
            code="ore_algebra.recurrence_ogf_differential_order",
            message="the recurrence coefficient degree exceeds the differential-operator order bound",
        )
    if work > MAX_RECURRENCE_OGF_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("recurrence",),
            code="ore_algebra.recurrence_ogf_work",
            message="recurrence to OGF conversion exceeds its admitted exact-arithmetic work",
        )
    values = [value.as_fraction() for value in request.initial_coefficients.values]
    # Only nonzero boundary products are materialized. A large initial value
    # annihilated by q_i(-i), or canceled by its rational denominator, contributes
    # no output coefficient and must not consume the carrier's numerator budget.
    forcing: _Poly = {}
    for shift, polynomial in polynomials:
        for index in range(shift):
            evaluation = _evaluate_polynomial(polynomial, index - shift)
            if evaluation and values[index]:
                degree = operator.order - shift + index
                forcing[degree] = forcing.get(degree, Fraction(0)) + values[index] * evaluation
                if not forcing[degree]:
                    del forcing[degree]
    boundary_numerator_bits = max(
        (abs(value.numerator).bit_length() for value in forcing.values()), default=1
    )
    # Operator coefficients from distinct shifts occupy distinct (D-order,
    # x-degree) slots. Boundary terms have already been combined by output
    # degree, so only their retained denominators belong in this bound.
    coefficient_bounds = _transformed_coefficient_bounds(polynomials, operator.order)
    numerator_bits = max((bound[0] for bound in coefficient_bounds), default=1)
    denominator_bits = max(
        max((bound[1] for bound in coefficient_bounds), default=1),
        max(
            (value.denominator.bit_length() for value in forcing.values()),
            default=1,
        ),
    )
    # Evaluating a degree-d polynomial at integer points of magnitude at most r
    # adds at most d*ceil(log2(r+1)) bits. The shift order alone does not imply
    # coefficient growth (e.g. a_(n+r)=0 has unit coefficients throughout).
    evaluation_growth_bits = maximum_degree * operator.order.bit_length()
    growth_bits = (
        max(numerator_bits, boundary_numerator_bits, denominator_bits)
        + evaluation_growth_bits
    )
    # Only nonzero accumulated boundary coefficients contribute output monomials.
    maximum_output_degree = max(maximum_output_degree, max(forcing, default=-1))
    if maximum_output_degree > MAX_SHIFT_COEFFICIENT_DEGREE:
        raise OperationResourceAdmissionError(
            location=("recurrence", "terms"),
            code="ore_algebra.recurrence_ogf_polynomial_degree",
            message="the cleared OGF equation exceeds the differential coefficient degree bound",
        )
    output_digits = _digits_for_bit_bound(growth_bits)
    if output_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.recurrence_ogf_coefficient_digits",
            message="the OGF differential equation may exceed the exact rational coefficient bound",
        )
    return polynomials, maximum_degree, forcing


def _expand_equation(
    polynomials: list[tuple[int, _Poly]],
    shift_order: int,
    maximum_degree: int,
    admitted_forcing: _Poly,
) -> tuple[DifferentialOreOperator, RationalFunction]:
    """Expand q_i(Theta-i), where Theta=xD, plus its initial polynomial."""
    stirling: list[list[int]] = [[1]]
    for power in range(1, maximum_degree + 1):
        previous = stirling[-1]
        row = [0] * (power + 1)
        for derivative_order in range(1, power + 1):
            same_degree = (
                previous[derivative_order] if derivative_order < len(previous) else 0
            )
            row[derivative_order] = (
                derivative_order * same_degree + previous[derivative_order - 1]
            )
        stirling.append(row)

    coefficients: dict[int, _Poly] = {}
    forcing = dict(admitted_forcing)
    for shift, polynomial in polynomials:
        for theta_power, scalar in _shift_polynomial(polynomial, -shift).items():
            for derivative_order in range(theta_power + 1):
                multiplier = stirling[theta_power][derivative_order]
                if multiplier:
                    degree = shift_order - shift + derivative_order
                    target = coefficients.setdefault(derivative_order, {})
                    target[degree] = (
                        target.get(degree, Fraction(0)) + scalar * multiplier
                    )
                    if not target[degree]:
                        del target[degree]

    differential_operator = DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": derivative_order,
                    "coefficient": _encode_rational_polynomial(polynomial, "x"),
                }
                for derivative_order, polynomial in sorted(coefficients.items())
                if polynomial
            ],
        }
    )
    return differential_operator, _encode_rational_polynomial(forcing, "x")


def polynomial_recurrence_to_ogf_equation(
    recurrence: ShiftOreOperator | Mapping[str, Any],
    initial_coefficients: FiniteRationalSequence | Mapping[str, Any],
) -> RecurrenceOGFEquation:
    r"""Return L(F)=B for a polynomial recurrence and its OGF.

    For ``sum_i q_i(n) a_(n+i)=0`` and ``F(x)=sum_n a_n x^n``, set
    ``L=sum_i x^(r-i)q_i(xD-i)`` and
    ``B=sum_i x^(r-i)q_i(xD-i)sum_(m<i)a_mx^m``, with ``r`` the largest shift.
    This transforms the relation; it does not verify an infinite sequence or
    assert analytic convergence.
    """
    request, operator = _canonical_request(recurrence, initial_coefficients)
    polynomials, maximum_degree, forcing = _admit_transform(request, operator)
    differential_operator, forcing = _expand_equation(
        polynomials,
        operator.order,
        maximum_degree,
        forcing,
    )
    return RecurrenceOGFEquation._from_kernel(
        operator,
        request.initial_coefficients,
        differential_operator,
        forcing,
    )
