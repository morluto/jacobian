"""Exact SymPy-backed polynomial computations over ``QQ``."""

from __future__ import annotations

from typing import Any

from jacobian.contracts.polynomial_operations import (
    PolynomialBezoutIdentity,
    PolynomialDiscriminantRequest,
    PolynomialDiscriminantResult,
    PolynomialGcdRequest,
    PolynomialGcdResult,
    PolynomialGroebnerBasisRequest,
    PolynomialGroebnerBasisResult,
    PolynomialInvariantValue,
    PolynomialResultantRequest,
    PolynomialResultantResult,
    PolynomialScalarValue,
    PolynomialSquareFreeDecompositionResult,
    PolynomialSquareFreeFactor,
    PolynomialSquareFreeRequest,
    PolynomialValue,
)
from jacobian.contracts.polynomials import RationalPolynomial
from jacobian.domains.polynomial import kernels
from jacobian.domains.polynomial.conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)

_MAX_OUTPUT_TERMS = 1024


class PolynomialOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _result_polynomial(poly: object, variables: tuple[str, ...]) -> RationalPolynomial:
    try:
        return rational_polynomial_from_sympy(
            poly,
            variables,
            maximum_terms=_MAX_OUTPUT_TERMS,
        )
    except ValueError as exc:
        if "term operation budget" in str(exc):
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
    left_multiplier, right_multiplier, gcd = kernels.polynomial_gcdex(left, right)
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
    value = kernels.polynomial_resultant(
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
    value = kernels.polynomial_discriminant(
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
        kernels.polynomial_square_free_decomposition(source)
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


def polynomial_groebner_basis(
    request: PolynomialGroebnerBasisRequest,
) -> PolynomialGroebnerBasisResult:
    """Compute one complete reduced basis inside the isolated worker."""

    variables = request.generators[0].variables
    wire_basis = tuple(
        _result_polynomial(polynomial, variables)
        for polynomial in kernels.polynomial_groebner_basis(
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
