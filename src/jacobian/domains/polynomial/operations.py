"""Exact SymPy-backed polynomial computations over ``QQ``."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.exact import CanonicalRational
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
from jacobian.contracts.polynomials import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_MAX_OUTPUT_TERMS = 1024


class PolynomialOutputBudgetError(RuntimeError):
    """A valid computation produced more output than its public contract permits."""


def _symbols(variables: tuple[str, ...]) -> tuple[Any, ...]:
    from sympy import Symbol

    return tuple(Symbol(variable) for variable in variables)


def _poly(polynomial: RationalPolynomial) -> Any:
    from sympy import QQ, Poly, Rational

    generators = _symbols(polynomial.variables)
    coefficients = {
        term.exponents: Rational(term.coefficient.as_fraction())
        for term in polynomial.polynomial.terms
    }
    return Poly.from_dict(coefficients, *generators, domain=QQ)


def _rational(value: Any) -> CanonicalRational:
    fraction = Fraction(value)
    return CanonicalRational(
        num=format_canonical_integer(fraction.numerator),
        den=format_canonical_integer(fraction.denominator),
    )


def _wire(poly: Any, variables: tuple[str, ...]) -> RationalPolynomial:
    terms = tuple(
        (exponents, coefficient)
        for exponents, coefficient in poly.terms()
        if coefficient != 0
    )
    if len(terms) > _MAX_OUTPUT_TERMS:
        raise PolynomialOutputBudgetError(
            "polynomial result exceeds the 1024-term output budget"
        )
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=_rational(coefficient),
                    exponents=tuple(int(exponent) for exponent in exponents),
                )
                for exponents, coefficient in terms
            )
        ),
    )


def _invariant_value(
    expression: Any,
    remaining_variables: tuple[str, ...],
) -> PolynomialInvariantValue:
    from sympy import QQ, Poly

    if not remaining_variables:
        return PolynomialScalarValue(value=_rational(expression))
    return PolynomialValue(
        value=_wire(
            Poly(expression, *_symbols(remaining_variables), domain=QQ),
            remaining_variables,
        )
    )


def polynomial_gcd(request: PolynomialGcdRequest) -> PolynomialGcdResult:
    left = _poly(request.left)
    right = _poly(request.right)
    left_multiplier, right_multiplier, gcd = left.gcdex(right)
    variables = request.left.variables
    return PolynomialGcdResult(
        gcd=_wire(gcd, variables),
        bezout=PolynomialBezoutIdentity(
            left_multiplier=_wire(left_multiplier, variables),
            right_multiplier=_wire(right_multiplier, variables),
        ),
    )


def polynomial_resultant(
    request: PolynomialResultantRequest,
) -> PolynomialResultantResult:
    from sympy import resultant

    variables = request.left.variables
    elimination_index = variables.index(request.elimination_variable)
    generator = _symbols(variables)[elimination_index]
    value = resultant(
        _poly(request.left).as_expr(), _poly(request.right).as_expr(), generator
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
    from sympy import discriminant

    variables = request.polynomial.variables
    variable_index = variables.index(request.variable)
    generator = _symbols(variables)[variable_index]
    value = discriminant(_poly(request.polynomial).as_expr(), generator)
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
    from sympy import QQ, Poly

    source = _poly(request.polynomial)
    coefficient, raw_factors = source.sqf_list()
    canonical_factors = tuple(
        (factor.monic(), int(multiplicity)) for factor, multiplicity in raw_factors
    )
    factors = tuple(
        PolynomialSquareFreeFactor(
            factor=_wire(factor, request.polynomial.variables),
            multiplicity=multiplicity,
        )
        for factor, multiplicity in sorted(canonical_factors, key=lambda item: item[1])
    )
    reconstructed = Poly(
        coefficient,
        *_symbols(request.polynomial.variables),
        domain=QQ,
    )
    for factor, multiplicity in canonical_factors:
        reconstructed *= factor**multiplicity
    if reconstructed != source:
        raise RuntimeError("SymPy square-free decomposition did not reconstruct input")
    return PolynomialSquareFreeDecompositionResult(
        coefficient=_rational(coefficient),
        factors=factors,
        reconstructed=_wire(reconstructed, request.polynomial.variables),
    )


def polynomial_groebner_basis(
    request: PolynomialGroebnerBasisRequest,
) -> PolynomialGroebnerBasisResult:
    """Compute one complete reduced basis inside the isolated worker."""

    from sympy import QQ, groebner

    variables = request.generators[0].variables
    basis = groebner(
        [_poly(generator).as_expr() for generator in request.generators],
        *_symbols(variables),
        order=request.monomial_order,
        domain=QQ,
    )
    wire_basis = tuple(_wire(polynomial, variables) for polynomial in basis.polys)
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
