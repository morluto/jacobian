"""Bounded exact exponential-generating-function recurrence transform."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import (
    MAX_DIFFERENTIAL_ORDER,
    MAX_DIFFERENTIAL_TERMS,
    MAX_SHIFT_COEFFICIENT_DEGREE,
    MAX_SHIFT_COEFFICIENT_DIGITS,
    MAX_SHIFT_COEFFICIENT_TERMS,
    DifferentialOreOperator,
    ShiftOreOperator,
)
from jacobian.math.ore_algebras.recurrence_to_egf._models import (
    RecurrenceEGFEquation,
    RecurrenceEGFEquationRequest,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    require_canonical_rational_function,
)

_Poly = dict[int, Fraction]
MAX_RECURRENCE_EGF_WORK_CELLS = 100_000
MAX_RECURRENCE_EGF_OUTPUT_BYTES = 2 * 1024 * 1024


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


def _encode_rational_polynomial(value: _Poly) -> RationalFunction:
    return RationalFunction(
        domain="QQ",
        variables=("x",),
        numerator=_encode_polynomial(value),
        denominator=_encode_polynomial({0: Fraction(1)}),
    )


def _digits_for_bit_bound(bits: int) -> int:
    # 0.30103 is a strict rational upper approximation for log10(2).
    return (bits * 30_103) // 100_000 + 1


def _canonical_recurrence(
    recurrence: ShiftOreOperator | Mapping[str, Any],
) -> ShiftOreOperator:
    try:
        request = RecurrenceEGFEquationRequest.model_validate(
            {
                "recurrence": recurrence.model_dump()
                if isinstance(recurrence, ShiftOreOperator)
                else recurrence
            }
        )
        operator = ShiftOreOperator.model_validate(request.recurrence.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("recurrence",),
            code="ore_algebra.recurrence_egf_request",
            message=(
                "the recurrence must be nonzero with polynomial coefficients in QQ[n]"
            ),
        ) from exc
    for index, term in enumerate(operator.terms):
        try:
            require_canonical_rational_function(
                term.coefficient,
                maximum_terms=MAX_SHIFT_COEFFICIENT_TERMS,
                maximum_exponent=MAX_SHIFT_COEFFICIENT_DEGREE,
                maximum_coefficient_digits=MAX_SHIFT_COEFFICIENT_DIGITS,
                label=f"recurrence coefficient {index}",
            )
        except Exception as exc:
            raise OperationDomainValidationError(
                location=("recurrence", "terms", index, "coefficient"),
                code="ore_algebra.recurrence_egf_coefficient",
                message="recurrence coefficients must be canonical bounded polynomials in QQ[n]",
            ) from exc
    return operator


def _admit_transform(
    operator: ShiftOreOperator,
) -> tuple[list[tuple[int, _Poly]], int, int]:
    """Admit polynomial expansion and differential result before expansion."""
    polynomials: list[tuple[int, _Poly]] = []
    maximum_degree = 0
    maximum_differential_order = 0
    input_scalars: list[Fraction] = []
    work = len(operator.terms)
    for term in operator.terms:
        numerator = _decode_polynomial(term.coefficient.numerator)
        denominator = _decode_polynomial(term.coefficient.denominator)
        if denominator != {0: Fraction(1)}:
            raise OperationDomainValidationError(
                location=("recurrence", "terms", term.exponent, "coefficient"),
                code="ore_algebra.recurrence_egf_polynomial_domain",
                message="exponential generating-function conversion requires coefficients in QQ[n]",
            )
        degree = max(numerator, default=0)
        maximum_degree = max(maximum_degree, degree)
        maximum_differential_order = max(
            maximum_differential_order, term.exponent + degree
        )
        input_scalars.extend(numerator.values())
        polynomials.append((term.exponent, numerator))
        # Includes each binomial/Stirling coefficient update and its accumulation.
        work += len(numerator) * (degree + 1) ** 2
    if maximum_differential_order > MAX_DIFFERENTIAL_ORDER:
        raise OperationResourceAdmissionError(
            location=("recurrence", "terms"),
            code="ore_algebra.recurrence_egf_differential_order",
            message="the recurrence requires a differential operator above the order bound",
        )
    if maximum_degree > MAX_SHIFT_COEFFICIENT_DEGREE:
        raise OperationResourceAdmissionError(
            location=("recurrence", "terms"),
            code="ore_algebra.recurrence_egf_polynomial_degree",
            message="an EGF differential coefficient exceeds the polynomial degree bound",
        )
    if maximum_differential_order + 1 > MAX_DIFFERENTIAL_TERMS:
        raise OperationResourceAdmissionError(
            location=("recurrence", "terms"),
            code="ore_algebra.recurrence_egf_differential_terms",
            message="the EGF differential operator exceeds its sparse term bound",
        )
    if work > MAX_RECURRENCE_EGF_WORK_CELLS:
        raise OperationResourceAdmissionError(
            location=("recurrence",),
            code="ore_algebra.recurrence_egf_work",
            message="recurrence to EGF conversion exceeds its admitted exact-arithmetic work",
        )
    denominator_bits = sum(value.denominator.bit_length() for value in input_scalars)
    numerator_bits = max(
        (abs(value.numerator).bit_length() for value in input_scalars), default=1
    )
    growth_bits = numerator_bits + denominator_bits + 32 + 8 * maximum_degree
    output_digits = _digits_for_bit_bound(growth_bits)
    if output_digits > MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.recurrence_egf_coefficient_digits",
            message="the EGF differential equation may exceed the exact rational coefficient bound",
        )
    # The result retains the canonical recurrence, then adds at most one ODE
    # term per derivative order with at most degree+1 polynomial monomials.
    recurrence_bytes = len(operator.model_dump_json().encode("utf-8"))
    estimated_bytes = (
        recurrence_bytes
        + 512
        + (maximum_differential_order + 1)
        * (512 + (maximum_degree + 1) * (2 * output_digits + 256))
    )
    if estimated_bytes > MAX_RECURRENCE_EGF_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("result",),
            code="ore_algebra.recurrence_egf_output_bytes",
            message="the EGF differential equation exceeds its serialized output bound",
        )
    return polynomials, maximum_degree, maximum_differential_order


def _stirling_second_kind(order: int) -> tuple[tuple[int, ...], ...]:
    rows = [[1]]
    for power in range(1, order + 1):
        previous = rows[-1]
        current = [0] * (power + 1)
        for degree in range(1, power + 1):
            same_degree = previous[degree] if degree < len(previous) else 0
            current[degree] = previous[degree - 1] + degree * same_degree
        rows.append(current)
    return tuple(tuple(row) for row in rows)


def _expand_egf_operator(
    polynomials: list[tuple[int, _Poly]], maximum_degree: int
) -> DifferentialOreOperator:
    """Map q_i(n)S^i to q_i(xD)D^i in differential normal form."""
    stirling = _stirling_second_kind(maximum_degree)
    differential_coefficients: dict[int, _Poly] = {}
    for shift, polynomial in polynomials:
        for theta_power, scalar in polynomial.items():
            for theta_degree in range(theta_power + 1):
                multiplier = stirling[theta_power][theta_degree]
                if not multiplier:
                    continue
                derivative_order = theta_degree + shift
                coefficient = differential_coefficients.setdefault(derivative_order, {})
                coefficient[theta_degree] = (
                    coefficient.get(theta_degree, Fraction(0)) + scalar * multiplier
                )
                if not coefficient[theta_degree]:
                    del coefficient[theta_degree]
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": derivative_order,
                    "coefficient": _encode_rational_polynomial(polynomial),
                }
                for derivative_order, polynomial in sorted(
                    differential_coefficients.items()
                )
                if polynomial
            ],
        }
    )


def polynomial_recurrence_to_egf_equation(
    recurrence: ShiftOreOperator | Mapping[str, Any],
) -> RecurrenceEGFEquation:
    r"""Convert sum_i q_i(n)a_(n+i)=0 to its exponential-GF equation.

    For ``E(x)=sum_n a_n*x^n/n!``, ``D^i E`` has coefficient ``a_(n+i)``
    in the basis ``x^n/n!``. The Euler operator ``Theta=xD`` acts on that
    basis by multiplication by ``n``. Therefore the transformed equation is
    ``sum_i q_i(Theta)D^i E=0``; no initial-term forcing is introduced.
    """
    operator = _canonical_recurrence(recurrence)
    polynomials, maximum_degree, _maximum_order = _admit_transform(operator)
    differential_operator = _expand_egf_operator(polynomials, maximum_degree)
    return RecurrenceEGFEquation._from_kernel(operator, differential_operator)
