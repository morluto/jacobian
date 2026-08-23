"""Exact SymPy-backed polynomial computations over ``QQ``."""

from __future__ import annotations

from typing import Any, Literal

import sympy

from jacobian.math import polynomials
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._models import (
    IdealMembershipRequest,
    IdealMembershipResult,
    IdealNormalFormResult,
    PolynomialBezoutIdentity,
    PolynomialDiscriminantRequest,
    PolynomialDiscriminantResult,
    PolynomialFactorizationResult,
    PolynomialFactorRequest,
    PolynomialGcdRequest,
    PolynomialGcdResult,
    PolynomialGroebnerBasisRequest,
    PolynomialGroebnerBasisResult,
    PolynomialInvariantValue,
    PolynomialIrreducibleFactor,
    PolynomialResultantRequest,
    PolynomialResultantResult,
    PolynomialScalarValue,
    PolynomialSquareFreeDecompositionResult,
    PolynomialSquareFreeFactor,
    PolynomialSquareFreeRequest,
    PolynomialValue,
)
from jacobian.math.polynomials._replay import (
    _ring_element,
    _sparse_ring,
    ReductionWorkLimitExceeded,
    budgeted_reduce,
)
from jacobian.math.polynomials.values import RationalPolynomial

_MAX_OUTPUT_TERMS = 1024


class PolynomialOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _result_polynomial(poly: object, variables: tuple[str, ...]) -> RationalPolynomial:
    from pydantic import ValidationError

    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=_MAX_OUTPUT_TERMS,
        )
    except ValidationError as exc:
        # Exponent cap (32_768), term representation, or canonical-rational
        # limits during result materialization are budget overflows, not host
        # exceptions.  Map every such validation failure to the typed outcome.
        raise PolynomialOutputBudgetError(str(exc)) from exc
    except ValueError as exc:
        if "term operation budget" in str(exc):
            raise PolynomialOutputBudgetError(str(exc)) from exc
        if "exponent" in str(exc).lower() or "representation limit" in str(exc).lower():
            raise PolynomialOutputBudgetError(str(exc)) from exc
        raise


def _invariant_value(
    expression: Any,
    remaining_variables: tuple[str, ...],
) -> PolynomialInvariantValue:
    from sympy import QQ, Poly

    if not remaining_variables:
        return PolynomialScalarValue(value=rational_from_sympy(expression))
    return PolynomialValue(
        value=_result_polynomial(
            Poly(expression, *symbols_for_variables(remaining_variables), domain=QQ),
            remaining_variables,
        )
    )


def polynomial_gcd(request: PolynomialGcdRequest) -> PolynomialGcdResult:
    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    left_multiplier, right_multiplier, gcd = polynomials.gcdex(left, right)
    variables = request.left.variables
    return PolynomialGcdResult(
        gcd=_result_polynomial(gcd, variables),
        bezout=PolynomialBezoutIdentity(
            left_multiplier=_result_polynomial(left_multiplier, variables),
            right_multiplier=_result_polynomial(right_multiplier, variables),
        ),
    )


def polynomial_resultant(
    request: PolynomialResultantRequest,
) -> PolynomialResultantResult:
    variables = request.left.variables
    elimination_index = variables.index(request.elimination_variable)
    generator = symbols_for_variables(variables)[elimination_index]
    value = polynomials.resultant(
        rational_polynomial_to_sympy(request.left),
        rational_polynomial_to_sympy(request.right),
        generator,
    )
    remaining_variables = tuple(
        variable for variable in variables if variable != request.elimination_variable
    )
    return PolynomialResultantResult(
        elimination_variable=request.elimination_variable,
        resultant=_invariant_value(value, remaining_variables),
    )


def polynomial_discriminant(
    request: PolynomialDiscriminantRequest,
) -> PolynomialDiscriminantResult:
    variables = request.polynomial.variables
    variable_index = variables.index(request.variable)
    generator = symbols_for_variables(variables)[variable_index]
    value = polynomials.discriminant(
        rational_polynomial_to_sympy(request.polynomial), generator
    )
    remaining_variables = tuple(
        variable for variable in variables if variable != request.variable
    )
    return PolynomialDiscriminantResult(
        variable=request.variable,
        discriminant=_invariant_value(value, remaining_variables),
    )


def polynomial_square_free_decomposition(
    request: PolynomialSquareFreeRequest,
) -> PolynomialSquareFreeDecompositionResult:
    source = rational_polynomial_to_sympy(request.polynomial)
    coefficient, canonical_factors, reconstructed = (
        polynomials.square_free_decomposition(source)
    )
    factors = tuple(
        PolynomialSquareFreeFactor(
            factor=_result_polynomial(factor, request.polynomial.variables),
            multiplicity=multiplicity,
        )
        for factor, multiplicity in sorted(canonical_factors, key=lambda item: item[1])
    )
    return PolynomialSquareFreeDecompositionResult(
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=_result_polynomial(reconstructed, request.polynomial.variables),
    )


def _irreducible_factor_sort_key(
    record: PolynomialIrreducibleFactor,
) -> tuple[int, int, tuple[tuple[tuple[int, ...], str, str], ...]]:
    return (
        record.multiplicity,
        max(
            (sum(term.exponents) for term in record.factor.polynomial.terms),
            default=0,
        ),
        tuple(
            (term.exponents, term.coefficient.num, term.coefficient.den)
            for term in record.factor.polynomial.terms
        ),
    )


def polynomial_factorization(
    request: PolynomialFactorRequest,
) -> PolynomialFactorizationResult:
    source = rational_polynomial_to_sympy(request.polynomial)
    coefficient, canonical_factors, reconstructed = polynomials.factorization(source)
    factors = tuple(
        sorted(
            (
                PolynomialIrreducibleFactor(
                    factor=_result_polynomial(factor, request.polynomial.variables),
                    multiplicity=multiplicity,
                )
                for factor, multiplicity in canonical_factors
            ),
            key=_irreducible_factor_sort_key,
        )
    )
    return PolynomialFactorizationResult(
        coefficient=rational_from_sympy(coefficient),
        factors=factors,
        reconstructed=_result_polynomial(reconstructed, request.polynomial.variables),
    )


def polynomial_groebner_basis(
    request: PolynomialGroebnerBasisRequest,
) -> PolynomialGroebnerBasisResult:
    """Compute one complete reduced basis inside the isolated worker."""

    variables = request.generators[0].variables
    wire_basis = tuple(
        _result_polynomial(polynomial, variables)
        for polynomial in polynomials.groebner_basis(
            tuple(
                rational_polynomial_to_sympy(generator)
                for generator in request.generators
            ),
            symbols_for_variables(variables),
            request.monomial_order,
        )
    )
    if len(wire_basis) > request.resource_budget.maximum_basis_polynomials:
        raise PolynomialOutputBudgetError(
            "Gröbner basis exceeds the requested polynomial-count limit"
        )
    if (
        sum(len(polynomial.polynomial.terms) for polynomial in wire_basis)
        > request.resource_budget.maximum_output_terms
    ):
        raise PolynomialOutputBudgetError(
            "Gröbner basis exceeds the requested aggregate term limit"
        )
    return PolynomialGroebnerBasisResult(
        variables=variables,
        monomial_order=request.monomial_order,
        basis=wire_basis,
    )


def _compute_membership_context(
    request: IdealMembershipRequest,
) -> tuple[tuple[RationalPolynomial, ...] | None, bool]:
    """Compute the source Gröbner basis and report the bounded outcome.

    The basis is wire-form when some bounded strategy concluded within
    every output budget.  ``None`` with ``exceeded`` reports evidenced
    complete-basis overflow for the typed budget outcome; ``None``
    without it reports that no strategy concluded, for the typed
    non-conclusion outcome.
    """

    from jacobian.math.polynomials._replay import incremental_source_groebner

    return incremental_source_groebner(request.ideal, request.monomial_order)


def _incomplete_normal_form(
    request: IdealMembershipRequest,
    status: Literal["BUDGET_EXCEEDED", "UNKNOWN"],
    groebner_basis: tuple[RationalPolynomial, ...] | None,
) -> IdealNormalFormResult:
    """Report a non-computed outcome without a partial exact artifact."""

    return IdealNormalFormResult(
        ideal=request.ideal,
        polynomial=request.polynomial,
        monomial_order=request.monomial_order,
        status=status,
        groebner_basis=groebner_basis,
        remainder=None,
    )


def polynomial_ideal_normal_form(
    request: IdealMembershipRequest,
) -> IdealNormalFormResult:
    """Reduce a polynomial modulo an ideal using a Groebner basis."""

    variables = request.ideal.variables
    symbols = symbols_for_variables(variables)
    wire_basis, exceeded = _compute_membership_context(request)
    if wire_basis is None:
        status: Literal["BUDGET_EXCEEDED", "UNKNOWN"] = (
            "BUDGET_EXCEEDED" if exceeded else "UNKNOWN"
        )
        return _incomplete_normal_form(request, status, None)
    # Reduce with a bounded sparse-ring division so an admitted request can
    # never expand an unbounded intermediate remainder before the 1,024-term
    # output boundary is noticed; overflow becomes the typed budget outcome.
    ring_context = _sparse_ring(variables, request.monomial_order)
    divisors = [_ring_element(ring_context, element) for element in wire_basis]
    dividend = _ring_element(ring_context, request.polynomial)
    try:
        replayed = budgeted_reduce(ring_context, dividend, divisors)
    except ReductionWorkLimitExceeded:
        # Work exhaustion without an overflowing artifact is a
        # non-conclusion, not evidenced overflow: the exact remainder may
        # sit well inside the output boundary while the naive reduction
        # order temporarily expands a larger intermediate.
        return _incomplete_normal_form(request, "UNKNOWN", None)
    if replayed is None:
        return _incomplete_normal_form(request, "BUDGET_EXCEEDED", wire_basis)
    try:
        remainder = _result_polynomial(
            sympy.Poly.from_dict(
                {
                    tuple(int(e) for e in monom): sympy.Rational(
                        int(coefficient.numerator),
                        int(coefficient.denominator),
                    )
                    for monom, coefficient in replayed.terms()
                },
                *symbols,
                domain=sympy.QQ,
            ),
            variables,
        )
    except (PolynomialOutputBudgetError, Exception) as exc:
        from pydantic import ValidationError

        if (
            isinstance(exc, (PolynomialOutputBudgetError, ValidationError))
            or "exponent" in str(exc).lower()
            or "representation limit" in str(exc).lower()
        ):
            return _incomplete_normal_form(request, "BUDGET_EXCEEDED", wire_basis)
        raise
    return IdealNormalFormResult(
        ideal=request.ideal,
        polynomial=request.polynomial,
        monomial_order=request.monomial_order,
        status="COMPUTED",
        groebner_basis=wire_basis,
        remainder=remainder,
    )


def polynomial_ideal_membership(
    request: IdealMembershipRequest,
) -> IdealMembershipResult:
    """Decide whether a polynomial lies in an ideal."""

    normal_form = polynomial_ideal_normal_form(request)
    if normal_form.status == "UNKNOWN":
        return IdealMembershipResult(
            ideal=normal_form.ideal,
            polynomial=normal_form.polynomial,
            monomial_order=normal_form.monomial_order,
            groebner_basis=None,
            status="UNKNOWN",
            normal_form=None,
        )
    if normal_form.status == "BUDGET_EXCEEDED":
        return IdealMembershipResult(
            ideal=normal_form.ideal,
            polynomial=normal_form.polynomial,
            monomial_order=normal_form.monomial_order,
            groebner_basis=normal_form.groebner_basis,
            status="BUDGET_EXCEEDED",
            normal_form=None,
        )
    remainder = normal_form.remainder
    if remainder is None:
        raise PolynomialOutputBudgetError(
            "COMPUTED normal form must carry its exact remainder"
        )
    is_zero = len(remainder.polynomial.terms) == 0
    return IdealMembershipResult(
        ideal=normal_form.ideal,
        polynomial=normal_form.polynomial,
        monomial_order=normal_form.monomial_order,
        groebner_basis=normal_form.groebner_basis,
        status="IN_IDEAL" if is_zero else "NOT_IN_IDEAL",
        normal_form=remainder,
    )
