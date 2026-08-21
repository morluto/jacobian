"""Domain functions for commutative algebra operations."""

from __future__ import annotations

import sympy

from jacobian.math.commutative_algebra_ops._models import (
    MAX_OUTPUT_TERMS,
    IdealQuotientRequest,
    IdealQuotientResult,
    IdealRadicalMembershipRequest,
    IdealRadicalMembershipResult,
    IdealRadicalRequest,
    IdealRadicalResult,
    IdealSaturationRequest,
    IdealSaturationResult,
)
from jacobian.math.commutative_algebra_ops._singular import (
    run_singular_ideal_operation,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_to_sympy,
    symbols_for_variables,
)


def compute_ideal_radical(request: IdealRadicalRequest) -> IdealRadicalResult:
    """Compute an exact ideal radical through the bounded Singular backend."""

    backend = run_singular_ideal_operation(
        "radical",
        request.ideal,
        None,
        request.resource_budget,
    )
    return IdealRadicalResult(
        outcome=backend.outcome,
        radical=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def compute_ideal_radical_membership(
    request: IdealRadicalMembershipRequest,
) -> IdealRadicalMembershipResult:
    """Decide radical membership by the exact Rabinowitsch criterion."""

    variable_symbols = symbols_for_variables(request.ideal.variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    polynomial = rational_polynomial_to_sympy(request.polynomial).as_expr()
    auxiliary = sympy.Dummy("jacobian_rabinowitsch")
    basis = sympy.groebner(
        [*ideal_generators, 1 - auxiliary * polynomial],
        *variable_symbols,
        auxiliary,
        order="grevlex",
        domain=sympy.QQ,
    )
    return IdealRadicalMembershipResult(in_radical=len(basis) == 1 and basis[0] == 1)


def compute_ideal_quotient(request: IdealQuotientRequest) -> IdealQuotientResult:
    """Compute an exact ideal quotient through the bounded Singular backend."""

    backend = run_singular_ideal_operation(
        "quotient",
        request.dividend,
        request.divisor,
        request.resource_budget,
    )
    return IdealQuotientResult(
        outcome=backend.outcome,
        quotient=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )




def compute_ideal_saturation(request: IdealSaturationRequest) -> IdealSaturationResult:
    """Compute an exact ideal saturation I : <d>^infinity through the bounded Singular backend."""

    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
        SparseRationalPolynomial,
    )
    saturation_ideal = RationalPolynomialIdeal(
        variables=request.ideal.variables,
        generators=(request.saturation_polynomial,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        request.ideal,
        saturation_ideal,
        request.resource_budget,
    )
    return IdealSaturationResult(
        outcome=backend.outcome,
        saturation=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )

__all__ = [
    "compute_ideal_quotient",
    "compute_ideal_radical",
    "compute_ideal_radical_membership",
    "compute_ideal_saturation",
]


def compute_groebner_basis(request: "GroebnerBasisRequest") -> "GroebnerBasisResult":
    """Compute a reduced Gröbner basis for a bounded ideal over QQ using SymPy."""
    from jacobian.math.commutative_algebra_ops._models import (
    MAX_OUTPUT_TERMS,
        GroebnerBasisResult,
    )
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
    from jacobian.math.polynomials.values import RationalPolynomial, RationalPolynomialIdeal, SparseRationalPolynomial

    variables = request.ideal.variables
    variable_symbols = symbols_for_variables(variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]

    order_map = {"lex": "lex", "grlex": "grlex", "grevlex": "grevlex"}
    order = order_map.get(request.monomial_order, "grevlex")

    basis = sympy.groebner(
        ideal_generators,
        *variable_symbols,
        order=order,
        domain=sympy.QQ,
    )

    basis_generators = tuple(
        rational_polynomial_from_sympy(
            sympy.Poly(expr, *variable_symbols, domain=sympy.QQ),
            variables,
            maximum_terms=MAX_OUTPUT_TERMS,
        )
        for expr in basis
    )

    ideal = RationalPolynomialIdeal(
        variables=variables,
        generators=basis_generators,
    )

    return GroebnerBasisResult(
        basis=ideal,
        generator_count=len(basis_generators),
        monomial_order=request.monomial_order,
    )


def compute_ideal_normal_form(request: "IdealNormalFormRequest") -> "IdealNormalFormResult":
    """Reduce one polynomial modulo an ideal using a Gröbner basis remainder."""
    from jacobian.math.commutative_algebra_ops._models import IdealNormalFormResult
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
    from jacobian.math.polynomials.values import SparseRationalPolynomial

    variables = request.ideal.variables
    variable_symbols = symbols_for_variables(variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]
    poly = rational_polynomial_to_sympy(request.polynomial).as_expr()

    _, remainder = sympy.reduced(
        poly,
        ideal_generators,
        *variable_symbols,
        order="grevlex",
        domain=sympy.QQ,
    )

    remainder_poly = rational_polynomial_from_sympy(
        sympy.Poly(remainder, *variable_symbols, domain=sympy.QQ),
        variables,
    )

    # The polynomial is in the ideal if and only if the remainder is zero
    in_ideal = len(remainder_poly.polynomial.terms) == 0

    return IdealNormalFormResult(
        remainder=remainder_poly,
        in_ideal=in_ideal,
    )


def compute_elimination_ideal(request: "EliminationIdealRequest") -> "EliminationIdealResult":
    """Compute the elimination ideal I ∩ QQ[remaining variables] using a lex Gröbner basis."""
    from jacobian.math.commutative_algebra_ops._models import EliminationIdealResult
    from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy
    from jacobian.math.polynomials.values import RationalPolynomial, RationalPolynomialIdeal, SparseRationalPolynomial

    variables = list(request.ideal.variables)
    eliminated_set = set(request.eliminated_variables)
    remaining = [v for v in variables if v not in eliminated_set]
    remaining_symbols = symbols_for_variables(tuple(remaining))

    variable_symbols = symbols_for_variables(request.ideal.variables)
    ideal_generators = [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in request.ideal.generators
    ]

    # Compute a lex Gröbner basis — the elimination ideal is generated by
    # the basis elements that only involve the remaining variables
    basis = sympy.groebner(
        ideal_generators,
        *variable_symbols,
        order="lex",
        domain=sympy.QQ,
    )

    elimination_generators = []
    for expr in basis:
        poly = sympy.Poly(expr, *variable_symbols, domain=sympy.QQ)
        # Check if this polynomial only involves the remaining variables
        involved_vars = set(str(s) for s in poly.free_symbols)
        if involved_vars and involved_vars.issubset(set(remaining)):
            elimination_generators.append(
                rational_polynomial_from_sympy(
                    sympy.Poly(expr, *remaining_symbols, domain=sympy.QQ),
                    tuple(remaining),
                )
            )

    if not elimination_generators:
        # The elimination ideal is the whole ring
        from jacobian._exact import CanonicalRational
        from jacobian.math.polynomials.values import RationalPolynomialTerm
        from jacobian.math.polynomials.values import RationalPolynomial as _RP
        one = _RP(
            variables=tuple(remaining),
            polynomial=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num="1", den="1"),
                        exponents=(0,) * len(remaining),
                    ),
                )
            ),
        )
        elimination_generators = [one]

    ideal = RationalPolynomialIdeal(
        variables=tuple(remaining),
        generators=tuple(elimination_generators),
    )

    return EliminationIdealResult(
        elimination_ideal=ideal,
        eliminated_variables=tuple(request.eliminated_variables),
    )
