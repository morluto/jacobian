"""Bounded exact right division for monic polynomial-coefficient ODE operators."""

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
from jacobian.math.ore_algebras.differential_right_division._models import (
    DifferentialRightDivisionRequest,
    DifferentialRightDivisionResult,
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


def _admit(request: DifferentialRightDivisionRequest) -> tuple[int, int, int]:
    """Preflight polynomial growth, scalar height, work and wire output."""
    encoded_bytes = len(request.model_dump_json().encode("utf-8"))
    if encoded_bytes > _MAX_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_right_division_input_bytes",
            message="the differential right-division request exceeds its byte bound",
        )

    operators = (request.dividend, request.divisor)
    for label, operator in zip(("dividend", "divisor"), operators, strict=True):
        for term in operator.terms:
            coefficient = term.coefficient
            if (
                len(coefficient.denominator.terms) != 1
                or coefficient.denominator.terms[0].exponents != (0,)
                or coefficient.denominator.terms[0].coefficient.as_fraction() != 1
            ):
                raise OperationDomainValidationError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.differential_right_division_polynomial_domain",
                    message="right division currently accepts integer polynomial coefficients in ZZ[x]",
                )
            values = tuple(
                poly_term.coefficient.as_fraction()
                for poly_term in coefficient.numerator.terms
            )
            if any(value.denominator != 1 for value in values):
                raise OperationDomainValidationError(
                    location=(label, "terms", term.order, "coefficient"),
                    code="ore_algebra.differential_right_division_integer_domain",
                    message="right division currently accepts integer polynomial coefficients in ZZ[x]",
                )
    divisor_leading = _poly_coefficient(request.divisor, request.divisor.order)
    if divisor_leading != {0: Fraction(1)}:
        raise OperationDomainValidationError(
            location=("divisor", "terms", request.divisor.order, "coefficient"),
            code="ore_algebra.differential_right_division_monic",
            message="the divisor must be monic in D (leading coefficient exactly one)",
        )

    # Identical operands have a forced one-step exact cancellation, regardless
    # of ambient coefficient degree/height. Avoid charging hypothetical growth
    # from multiplying the divisor by itself when the result is exactly 1, 0.
    if request.dividend == request.divisor:
        result_bytes = encoded_bytes + 1024
        if result_bytes > _MAX_OUTPUT_BYTES:
            raise OperationResourceAdmissionError(
                location=("request",),
                code="ore_algebra.differential_right_division_bound",
                message="right-division exact output exceeds its envelope",
            )
        return 0, 1, 1

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
    # A cancellation step forms at most 3*7*17 products into any output
    # coefficient. Two extra digits cover derivatives and binomial factors.
    result_digits = initial_digits + iterations * (initial_digits + 5)
    work = iterations * 7 * len(request.divisor.terms) * (result_degree + 1) * 8
    output_order = max(request.dividend.order, request.divisor.order)
    output_terms = 2 * (output_order + 1) * (result_degree + 1)
    result_bytes = encoded_bytes + output_terms * (160 + 2 * result_digits) + 1024
    if (
        result_degree > 64
        or result_digits > 64
        or work > _MAX_WORK
        or result_bytes > _MAX_OUTPUT_BYTES
    ):
        raise OperationResourceAdmissionError(
            location=("request",),
            code="ore_algebra.differential_right_division_bound",
            message="right-division coefficient growth, exact work, or output exceeds its envelope",
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


def differential_operator_right_divide_monic(
    dividend: DifferentialOreOperator | Mapping[str, Any],
    divisor: DifferentialOreOperator | Mapping[str, Any],
) -> DifferentialRightDivisionResult:
    """Return Q,R with ``dividend = Q*divisor + remainder`` and lower order.

    The divisor must be monic and both operators must have integer polynomial
    coefficients in ZZ[x]. Over the coefficient field QQ(x), the quotient and
    remainder are unique. Monicity keeps every exact intermediate in ZZ[x].
    """
    try:
        request = DifferentialRightDivisionRequest.model_validate(
            {
                "dividend": dividend.model_dump()
                if isinstance(dividend, DifferentialOreOperator)
                else dividend,
                "divisor": divisor.model_dump()
                if isinstance(divisor, DifferentialOreOperator)
                else divisor,
            }
        )
        request = DifferentialRightDivisionRequest.model_validate(request.model_dump())
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="ore_algebra.differential_right_division_request",
            message="right division requires a typed dividend and nonzero divisor",
        ) from exc

    # Reject out-of-domain requests and establish this operation's own
    # inexpensive structural bounds before shared canonicalization (which may
    # perform rational-function GCD work).
    _admit(request)
    left = _admit_differential_operator(_as_differential_operator(request.dividend))
    right = _admit_differential_operator(_as_differential_operator(request.divisor))
    request = DifferentialRightDivisionRequest.model_construct(
        dividend=left, divisor=right
    )
    _admit(request)

    remainder = {term.order: _poly_coefficient(left, term.order) for term in left.terms}
    divisor_polynomials = {
        term.order: _poly_coefficient(right, term.order) for term in right.terms
    }
    quotient: dict[int, _Poly] = {}
    divisor_order = right.order
    while remainder and max(remainder) >= divisor_order:
        remainder_order = max(remainder)
        shift = remainder_order - divisor_order
        leading = remainder[remainder_order]
        quotient[shift] = _poly_add(quotient.get(shift, {}), leading)
        for divisor_term_order, coefficient in divisor_polynomials.items():
            for derivative_order in range(shift + 1):
                differentiated = coefficient
                for _ in range(derivative_order):
                    differentiated = _differentiate(differentiated)
                if not differentiated:
                    continue
                target = shift - derivative_order + divisor_term_order
                contribution = _scale(
                    _poly_mul(leading, differentiated),
                    comb(shift, derivative_order),
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
    return DifferentialRightDivisionResult._from_kernel(
        left,
        right,
        quotient_value,
        remainder_value,
    )


__all__ = ["differential_operator_right_divide_monic"]
