"""Bounded exact left division for monic polynomial-coefficient ODE operators."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import comb
from typing import Any

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.ore_algebras._models import DifferentialOreOperator
from jacobian.math.ore_algebras.differential_left_division._models import (
    DifferentialLeftDivisionRequest,
    DifferentialLeftDivisionResult,
)
from jacobian.math.ore_algebras.operations import (
    _admit_differential_operator,
    _as_differential_operator,
    _decode_rf,
    _encode_differential_rf,
    _Poly,
    _poly_add,
    _poly_mul,
)

_MAX_INPUT_ORDER = 4
_MAX_INPUT_POLYNOMIAL_DEGREE = 2
_MAX_INPUT_POLYNOMIAL_TERMS = 3
_MAX_INPUT_SCALAR_DIGITS = 2
_MAX_WORK = 1_000_000
_MAX_OUTPUT_BYTES = 1_000_000


def _poly_coefficient(operator: DifferentialOreOperator, order: int) -> _Poly:
    term = next((term for term in operator.terms if term.order == order), None)
    if term is None:
        return {}
    numerator, denominator = _decode_rf(term.coefficient)
    if denominator != {0: Fraction(1)}:
        raise AssertionError("admitted polynomial coefficient acquired a denominator")
    return numerator


def _degree(polynomial: _Poly) -> int:
    return max(polynomial, default=-1)


def _coefficient_digits(operator: DifferentialOreOperator) -> int:
    return max(
        (
            max(len(str(abs(value.numerator))), len(str(value.denominator)))
            for term in operator.terms
            for coefficient in term.coefficient.numerator.terms
            for value in (coefficient.coefficient.as_fraction(),)
        ),
        default=1,
    )


def _admit(request: DifferentialLeftDivisionRequest) -> tuple[int, int, int]:
    """Preflight polynomial growth, scalar height, work and wire output."""
    encoded_bytes = len(request.model_dump_json().encode("utf-8"))
    if encoded_bytes > _MAX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_left_division_input_bytes",
            message="the differential left-division request exceeds its byte bound",
        )

    operators = (request.dividend, request.divisor)
    for label, operator in zip(("dividend", "divisor"), operators, strict=True):
        if operator.order > _MAX_INPUT_ORDER:
            raise OperationResourceAdmissionError(
                location=(label,),
                code="ore_algebra.differential_left_division_order",
                message="left division accepts operator order at most four",
            )
        for term in operator.terms:
            coefficient = term.coefficient
            if (
                len(coefficient.denominator.terms) != 1
                or coefficient.denominator.terms[0].exponents != (0,)
                or coefficient.denominator.terms[0].coefficient.as_fraction() != 1
            ):
                raise OperationDomainValidationError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.differential_left_division_polynomial_domain",
                    message="left division currently accepts integer polynomial coefficients in ZZ[x]",
                )
            values = tuple(
                poly_term.coefficient.as_fraction()
                for poly_term in coefficient.numerator.terms
            )
            if any(value.denominator != 1 for value in values):
                raise OperationDomainValidationError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.differential_left_division_integer_domain",
                    message="left division currently accepts integer polynomial coefficients in ZZ[x]",
                )
            if (
                len(values) > _MAX_INPUT_POLYNOMIAL_TERMS
                or any(
                    max(
                        len(str(abs(value.numerator))),
                        len(str(value.denominator)),
                    )
                    > _MAX_INPUT_SCALAR_DIGITS
                    for value in values
                )
                or any(
                    poly_term.exponents[0] > _MAX_INPUT_POLYNOMIAL_DEGREE
                    for poly_term in coefficient.numerator.terms
                )
            ):
                raise OperationResourceAdmissionError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.differential_left_division_coefficient_bound",
                    message=(
                        "left division accepts integer polynomial coefficients with degree "
                        "at most two, three terms, and two-digit rational scalars"
                    ),
                )

    divisor_leading = _poly_coefficient(request.divisor, request.divisor.order)
    if divisor_leading != {0: Fraction(1)}:
        raise OperationDomainValidationError(
            location=("divisor", "terms", request.divisor.order, "coefficient"),
            code="ore_algebra.differential_left_division_monic",
            message="the divisor must be monic in D (leading coefficient exactly one)",
        )

    iterations = max(0, request.dividend.order - request.divisor.order + 1)
    initial_degree = max(
        (
            _degree(_poly_coefficient(operator, term.order))
            for operator in operators
            for term in operator.terms
        ),
        default=0,
    )
    divisor_degree = max(
        (
            _degree(_poly_coefficient(request.divisor, term.order))
            for term in request.divisor.terms
        ),
        default=0,
    )
    result_degree = initial_degree + iterations * divisor_degree
    initial_digits = max(_coefficient_digits(operator) for operator in operators)
    # Each quotient derivative has order at most the divisor order. Bound its
    # coefficient growth by multiplying by (degree+1)^order, then allow for
    # the divisor coefficient, binomial factor, and at most 15 summed terms.
    derivative_digits = request.divisor.order * len(str(result_degree + 1))
    result_digits = initial_digits + iterations * (
        initial_digits + derivative_digits + 3
    )
    # Per divisor term: at most five quotient derivatives, each multiplied by
    # a three-term coefficient, added into a degree-bounded residual; derivative
    # construction is also bounded by five sparse polynomial passes.
    work = iterations * len(request.divisor.terms) * 56 * (result_degree + 1)
    output_terms = 2 * (_MAX_INPUT_ORDER + 1) * (result_degree + 1)
    result_bytes = encoded_bytes + output_terms * (512 + 2 * result_digits) + 1024
    if (
        result_degree > 64
        or result_digits > 64
        or work > _MAX_WORK
        or result_bytes > _MAX_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_left_division_bound",
            message="left-division coefficient growth, exact work, or output exceeds its envelope",
        )
    return result_degree, result_digits, work


def _differentiate(polynomial: _Poly) -> _Poly:
    return {
        exponent - 1: exponent * value
        for exponent, value in polynomial.items()
        if exponent
    }


def _scale(polynomial: _Poly, scalar: int) -> _Poly:
    return {exponent: scalar * value for exponent, value in polynomial.items()}


def _operator_from_polynomials(values: Mapping[int, _Poly]) -> DifferentialOreOperator:
    return DifferentialOreOperator.model_validate(
        {
            "variable": "x",
            "terms": [
                {
                    "order": order,
                    "coefficient": _encode_differential_rf(
                        (polynomial, {0: Fraction(1)})
                    ),
                }
                for order, polynomial in sorted(values.items())
                if polynomial
            ],
        }
    )


def differential_operator_left_divide_monic(
    dividend: DifferentialOreOperator | Mapping[str, Any],
    divisor: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialLeftDivisionResult:
    """Return Q,R with ``dividend = divisor*Q + remainder`` and lower order.

    The divisor must be monic and both operators must have integer polynomial
    coefficients in ZZ[x]. Over the coefficient field QQ(x), the quotient and
    remainder are unique. Monicity keeps every exact intermediate in ZZ[x].
    """
    try:
        request = DifferentialLeftDivisionRequest.model_validate(
            {
                "dividend": dividend.model_dump()
                if isinstance(dividend, DifferentialOreOperator)
                else dividend,
                "divisor": divisor.model_dump()
                if isinstance(divisor, DifferentialOreOperator)
                else divisor,
            }
        )
        request = DifferentialLeftDivisionRequest.model_validate(request.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.differential_left_division_request",
            message="left division requires a typed dividend and nonzero divisor",
        ) from exc

    dividend_value = _admit_differential_operator(
        _as_differential_operator(request.dividend)
    )
    divisor_value = _admit_differential_operator(
        _as_differential_operator(request.divisor)
    )
    request = DifferentialLeftDivisionRequest.model_construct(
        dividend=dividend_value, divisor=divisor_value
    )
    _admit(request)

    # For A = B*Q + R, cancel the current leading term with q*D^shift
    # on the right of B. The Weyl rule differentiates q, not B's coefficient.
    remainder = {
        term.order: _poly_coefficient(dividend_value, term.order)
        for term in dividend_value.terms
    }
    divisor_polynomials = {
        term.order: _poly_coefficient(divisor_value, term.order)
        for term in divisor_value.terms
    }
    quotient: dict[int, _Poly] = {}
    divisor_order = divisor_value.order
    while remainder and max(remainder) >= divisor_order:
        remainder_order = max(remainder)
        shift = remainder_order - divisor_order
        quotient_coefficient = remainder[remainder_order]
        quotient[shift] = _poly_add(quotient.get(shift, {}), quotient_coefficient)
        for divisor_term_order, divisor_coefficient in divisor_polynomials.items():
            differentiated_quotient = quotient_coefficient
            for derivative_order in range(divisor_term_order + 1):
                if derivative_order:
                    differentiated_quotient = _differentiate(differentiated_quotient)
                if not differentiated_quotient:
                    break
                target = shift + divisor_term_order - derivative_order
                contribution = _scale(
                    _poly_mul(divisor_coefficient, differentiated_quotient),
                    comb(divisor_term_order, derivative_order),
                )
                updated = _poly_add(remainder.get(target, {}), _scale(contribution, -1))
                if updated:
                    remainder[target] = updated
                else:
                    remainder.pop(target, None)
        if remainder.get(remainder_order):
            raise AssertionError(
                "monic cancellation did not lower the differential order"
            )

    quotient_value = _operator_from_polynomials(quotient)
    remainder_value = _operator_from_polynomials(remainder)
    return DifferentialLeftDivisionResult._from_kernel(
        dividend_value,
        divisor_value,
        quotient_value,
        remainder_value,
    )


__all__ = ["differential_operator_left_divide_monic"]
