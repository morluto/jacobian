"""Domain functions for commutative algebra operations."""

from __future__ import annotations

from fractions import Fraction

import sympy

from jacobian.math.commutative_algebra_ops._models import (
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


def _groebner_signature(variables, expressions) -> tuple:
    """The canonical reduced Groebner basis of the spanned ideal.

    Two finite presentations of the same ideal produce equal signatures,
    so a replayed relation can be compared against any claimed
    presentation without depending on generator ordering.
    """
    from fractions import Fraction

    if not expressions:
        return ()
    basis = sympy.groebner(expressions, *variables, order="lex")
    signature = []
    for expr in basis.exprs:
        component = sympy.Poly(expr, *variables, domain="QQ")
        terms = tuple(
            (
                monomial,
                Fraction(int(coefficient.numerator), int(coefficient.denominator)),
            )
            for monomial, coefficient in sorted(
                zip(component.monoms(), component.coeffs(), strict=True), reverse=True
            )
        )
        signature.append(terms)
    return tuple(sorted(signature))


def replay_saturation(request: IdealSaturationRequest) -> tuple:
    """Recompute I : <d>^infinity exactly and return its Groebner signature."""
    from sympy import Symbol

    variables = symbols_for_variables(request.ideal.variables)
    t = Symbol("_saturation_t")
    polys = [
        *[
            rational_polynomial_to_sympy(generator).as_expr()
            for generator in request.ideal.generators
        ],
        t * rational_polynomial_to_sympy(request.saturation_polynomial).as_expr() - 1,
    ]
    elimination = sympy.groebner(polys, t, *variables, order="lex")
    saturated = [expr for expr in elimination.exprs if not expr.has(t)]
    if not saturated:
        # The saturation is the whole ring; its reduced basis is "1".
        return ((((), Fraction(1, 1)),),)
    return _groebner_signature(variables, saturated)


def compute_ideal_saturation(request: IdealSaturationRequest) -> IdealSaturationResult:
    """Compute an exact ideal saturation I : <d>^infinity through the bounded Singular backend."""

    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
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
        request=request,
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


def rational_expressions_of_ideal(ideal) -> list:
    """Convert one wire ideal into SymPy expressions over its own ring."""
    return [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in ideal.generators
    ]
