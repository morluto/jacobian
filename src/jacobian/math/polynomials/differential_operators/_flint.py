"""Private python-flint adapter for exact differential-operator application."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import format_canonical_integer
from jacobian.math.polynomials.differential_operators._bounds import (
    MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
    MAX_APPLICATION_OUTPUT_TERMS,
    ApplicationEnvelope,
    _is_identity_operator,
)
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _context(variables: tuple[str, ...]) -> Any:
    from flint import fmpq_mpoly_ctx

    active = fmpq_mpoly_ctx.get(variables, "lex")
    if tuple(active.names()) != variables:
        raise RuntimeError("python-flint did not preserve the ordered variable axis")
    return active


def _coefficient(value: CanonicalRational) -> Any:
    from flint import fmpq

    numerator, denominator = value.as_integer_ratio()
    return fmpq(numerator, denominator)


def _polynomial_to_backend(polynomial: RationalPolynomial, active: Any) -> Any:
    return active.from_dict(
        {
            term.exponents: _coefficient(term.coefficient)
            for term in polynomial.polynomial.terms
        }
    )


def _operator_to_backend(
    operator: ConstantCoefficientDifferentialOperator,
    active: Any,
) -> Any:
    return active.from_dict(
        {term.orders: _coefficient(term.coefficient) for term in operator.terms}
    )


def _zero_polynomial(variables: tuple[str, ...]) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(),
    )


def _power_operator(operator: Any, iterations: int, active: Any) -> Any:
    """Power one sparse operator with the schedule bounded during admission."""

    powered = active.from_dict({(0,) * active.nvars(): 1})
    base = operator
    remaining = iterations
    while remaining:
        if remaining & 1:
            powered = powered * base
        remaining >>= 1
        if remaining:
            base = base * base
    return powered


def _polynomial_from_backend(
    polynomial: Any,
    variables: tuple[str, ...],
    *,
    candidate_term_bound: int,
) -> RationalPolynomial:
    if tuple(polynomial.context().names()) != variables:
        raise RuntimeError("python-flint result changed the ordered variable axis")
    terms = tuple(polynomial.to_dict().items())
    if len(terms) > min(candidate_term_bound, MAX_APPLICATION_OUTPUT_TERMS):
        raise RuntimeError("python-flint result exceeded the admitted support bound")

    wire_terms: list[RationalPolynomialTerm] = []
    for exponents, coefficient in sorted(terms, reverse=True):
        rational = CanonicalRational(
            num=format_canonical_integer(int(coefficient.numerator)),
            den=format_canonical_integer(int(coefficient.denominator)),
        )
        require_bounded_rational(
            rational,
            max_digits=MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
            label="differential-operator result coefficient",
        )
        wire_terms.append(
            RationalPolynomialTerm(
                coefficient=rational,
                exponents=tuple(int(exponent) for exponent in exponents),
            )
        )
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(terms=tuple(wire_terms)),
    )


def _rescaled_source(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
) -> RationalPolynomial:
    """Scale the source by the zero-order coefficient raised to ``iterations``."""

    zero_coefficient = next(
        (
            term.coefficient.as_fraction()
            for term in operator.terms
            if not any(term.orders)
        ),
        Fraction(0),
    )
    if zero_coefficient == 1:
        scaled = Fraction(1)
    elif zero_coefficient == -1:
        scaled = Fraction(-1 if iterations % 2 else 1)
    else:
        scaled = zero_coefficient**iterations
    wire_terms = []
    for term in polynomial.polynomial.terms:
        value = term.coefficient.as_fraction() * scaled
        if value == 0:
            continue
        rational = CanonicalRational(
            num=format_canonical_integer(value.numerator),
            den=format_canonical_integer(value.denominator),
        )
        require_bounded_rational(
            rational,
            max_digits=MAX_APPLICATION_OUTPUT_COEFFICIENT_DIGITS,
            label="differential-operator result coefficient",
        )
        wire_terms.append(
            RationalPolynomialTerm(
                coefficient=rational,
                exponents=term.exponents,
            )
        )
    return RationalPolynomial(
        variables=polynomial.variables,
        polynomial=SparseRationalPolynomial(terms=tuple(wire_terms)),
    )


def apply_with_flint(
    polynomial: RationalPolynomial,
    operator: ConstantCoefficientDifferentialOperator,
    iterations: int,
    envelope: ApplicationEnvelope,
) -> RationalPolynomial:
    """Return ``operator**iterations`` applied to ``polynomial`` exactly."""

    if envelope.guaranteed_zero:
        return _zero_polynomial(polynomial.variables)
    if envelope.rescale_only:
        return _rescaled_source(polynomial, operator, iterations)
    if iterations == 0 or _is_identity_operator(operator):
        return polynomial

    active = _context(polynomial.variables)
    source = _polynomial_to_backend(polynomial, active)
    powered_operator = _power_operator(
        _operator_to_backend(operator, active),
        iterations,
        active,
    )
    operator_terms = tuple(powered_operator.to_dict().items())
    if len(operator_terms) > envelope.expanded_operator_terms:
        raise RuntimeError("python-flint operator power exceeded its support bound")

    maximum_exponents = tuple(
        max(
            (term.exponents[index] for term in polynomial.polynomial.terms),
            default=0,
        )
        for index in range(len(polynomial.variables))
    )
    maximum_total_degree = max(
        (sum(term.exponents) for term in polynomial.polynomial.terms),
        default=0,
    )
    result = active.from_dict({})
    for orders, coefficient in operator_terms:
        derivative_orders = tuple(int(order) for order in orders)
        if sum(derivative_orders) > maximum_total_degree:
            continue
        if any(
            order > maximum
            for order, maximum in zip(
                derivative_orders,
                maximum_exponents,
                strict=True,
            )
        ):
            continue
        differentiated = source
        for variable_index, order in enumerate(derivative_orders):
            for _ in range(order):
                differentiated = differentiated.derivative(variable_index)
        if not differentiated.is_zero():
            result = result + coefficient * differentiated

    return _polynomial_from_backend(
        result,
        polynomial.variables,
        candidate_term_bound=envelope.candidate_output_terms,
    )


__all__ = ["apply_with_flint"]
