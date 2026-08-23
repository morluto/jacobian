"""Domain-owned multivariate polynomial operations over ``QQ``."""

from __future__ import annotations

from typing import Any

from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.multivariate._models import (
    _MAX_SUBRESULTANT_SEQUENCE_TERMS,
    MultivariateDivisionRequest,
    MultivariateDivisionResult,
    MultivariateGcdRequest,
    MultivariateGcdResult,
    MultivariatePolynomialValue,
    MultivariatePrincipalSubresultantCoefficient,
    MultivariateResultantRequest,
    MultivariateResultantResult,
    MultivariateScalarValue,
    MultivariateSubresultantMember,
    MultivariateSubresultantSequenceRequest,
    MultivariateSubresultantSequenceResult,
    _degree_in_variable,
)

_MAX_OUTPUT_TERMS = 1024


class MultivariateOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _result_polynomial(
    poly: Any,
    variables: tuple[str, ...],
) -> Any:
    """Convert a SymPy ``Poly`` to a ``RationalPolynomial``, re-raising budget errors."""

    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=_MAX_OUTPUT_TERMS,
        )
    except ValueError as exc:
        if "term operation budget" in str(exc):
            raise MultivariateOutputBudgetError(str(exc)) from exc
        raise


def compute_multivariate_gcd(request: MultivariateGcdRequest) -> MultivariateGcdResult:
    """Compute the GCD of two multivariate polynomials over ``QQ``."""

    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    gcd = left.gcd(right)
    return MultivariateGcdResult(
        gcd=_result_polynomial(gcd, request.left.variables),
    )


def compute_multivariate_division(
    request: MultivariateDivisionRequest,
) -> MultivariateDivisionResult:
    """Divide one multivariate polynomial by another with a declared monomial order."""

    variables = request.left.variables
    symbols = symbols_for_variables(variables)

    # Build a low-level ring with the requested monomial order so that
    # multivariate division respects the declared term ordering.  The
    # ``Poly`` API uses the implicit lex order and does not expose the
    # monomial-order keyword in its constructor.
    from sympy import QQ
    from sympy.polys.rings import ring as sympy_ring

    ring_obj = sympy_ring(symbols, QQ, request.monomial_order)[0]

    left_poly = rational_polynomial_to_sympy(request.left)
    right_poly = rational_polynomial_to_sympy(request.right)

    left_ring = ring_obj.from_dict(
        {exponents: coeff for exponents, coeff in left_poly.terms()}  # noqa: C416
    )
    right_ring = ring_obj.from_dict(
        {exponents: coeff for exponents, coeff in right_poly.terms()}  # noqa: C416
    )

    quotient_ring, remainder_ring = left_ring.div(right_ring)

    # Convert back to ``Poly`` for the canonical sparse wire contract.
    quotient_poly = _to_poly(quotient_ring, symbols)
    remainder_poly = _to_poly(remainder_ring, symbols)

    # Verify the exact reconstruction: left == quotient * right + remainder.
    reconstruction = quotient_poly * right_poly + remainder_poly
    if reconstruction != left_poly:
        raise MultivariateOutputBudgetError(
            "multivariate division reconstruction failed"
        )

    return MultivariateDivisionResult(
        quotient=_result_polynomial(quotient_poly, variables),
        remainder=_result_polynomial(remainder_poly, variables),
        monomial_order=request.monomial_order,
    )


def compute_multivariate_resultant(
    request: MultivariateResultantRequest,
) -> MultivariateResultantResult:
    """Compute the resultant of two multivariate polynomials w.r.t. one variable."""

    from sympy import QQ, Poly

    from jacobian.math.polynomials._sympy import polynomial_resultant

    variables = request.left.variables
    elimination_index = variables.index(request.elimination_variable)
    generator = symbols_for_variables(variables)[elimination_index]

    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    value = polynomial_resultant(left, right, generator)

    remaining_variables = tuple(
        variable for variable in variables if variable != request.elimination_variable
    )
    if not remaining_variables:
        # The resultant is a rational scalar.
        from jacobian.math.polynomials._conversions import rational_from_sympy

        return MultivariateResultantResult(
            elimination_variable=request.elimination_variable,
            resultant=MultivariateScalarValue(
                value=rational_from_sympy(value),
            ),
        )
    resultant_poly = Poly(value, *symbols_for_variables(remaining_variables), domain=QQ)
    return MultivariateResultantResult(
        elimination_variable=request.elimination_variable,
        resultant=MultivariatePolynomialValue(
            value=_result_polynomial(resultant_poly, remaining_variables),
        ),
    )


def compute_multivariate_subresultant_sequence(
    request: MultivariateSubresultantSequenceRequest,
) -> MultivariateSubresultantSequenceResult:
    """Return the exact nonzero Brown PRS in one declared main variable."""

    from jacobian.math.polynomials.multivariate._subresultants import (
        polynomial_leading_coefficient_in_remaining_ring,
        polynomial_resultant_in_remaining_ring,
        polynomial_subresultant_sequence,
    )

    source_order, polynomials, principal_coefficients = (
        polynomial_subresultant_sequence(
            request.left,
            request.right,
            request.main_variable,
            maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
        )
    )
    variable_index = request.left.variables.index(request.main_variable)
    members = tuple(
        MultivariateSubresultantMember(
            polynomial=polynomial,
            degree_in_main_variable=_degree_in_variable(polynomial, variable_index),
        )
        for polynomial in polynomials
    )
    degrees = {member.degree_in_main_variable for member in members}
    greatest_degree = max(degrees)
    resultant = polynomial_resultant_in_remaining_ring(
        request.left,
        request.right,
        request.main_variable,
        maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
    )
    gcd_degree = members[-1].degree_in_main_variable
    gcd_leading_coefficient = polynomial_leading_coefficient_in_remaining_ring(
        members[-1].polynomial,
        request.main_variable,
        maximum_terms=_MAX_SUBRESULTANT_SEQUENCE_TERMS,
    )
    left_degree = _degree_in_variable(request.left, variable_index)
    right_degree = _degree_in_variable(request.right, variable_index)
    return MultivariateSubresultantSequenceResult(
        left=request.left,
        right=request.right,
        main_variable=request.main_variable,
        source_order=source_order,
        members=members,
        skipped_member_degrees=tuple(
            degree for degree in range(greatest_degree + 1) if degree not in degrees
        ),
        principal_subresultant_coefficients=tuple(
            MultivariatePrincipalSubresultantCoefficient(
                index=index,
                coefficient=coefficient,
            )
            for index, coefficient in enumerate(principal_coefficients)
        ),
        resultant=resultant,
        resultant_sign_from_sequence_order=(
            -1
            if source_order == "RIGHT_LEFT" and left_degree * right_degree % 2 == 1
            else 1
        ),
        gcd_member_index=len(members) - 1,
        gcd_degree_in_main_variable=gcd_degree,
        gcd_member_leading_coefficient=gcd_leading_coefficient,
    )


def _to_poly(ring_element: Any, symbols: tuple[Any, ...]) -> Any:
    """Convert a low-level ring element to a SymPy ``Poly`` in ``QQ``."""

    from sympy import QQ, Poly

    return Poly(ring_element.as_expr(), *symbols, domain=QQ)
