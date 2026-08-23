"""Domain functions for commutative algebra operations."""

from __future__ import annotations

from typing import Any

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
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
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
        request=request,
        outcome=backend.outcome,
        saturation=backend.ideal,
        backend_version=backend.backend_version,
        detail=backend.detail,
    )


def _sympy_generators(ideal: RationalPolynomialIdeal) -> list[Any]:
    return [
        rational_polynomial_to_sympy(generator).as_expr()
        for generator in ideal.generators
    ]


def _reduces_to_zero(basis: Any, expr: Any) -> bool:
    remainder = basis.reduce(expr)[1]
    return bool(remainder == 0)


def _mutually_contained(
    left_gens: list[Any], right_gens: list[Any], symbols: tuple[Any, ...]
) -> bool:
    """Mutual generator containment of the two ideals."""
    left_basis = sympy.groebner(left_gens, *symbols, order="grevlex", domain=sympy.QQ)
    right_basis = sympy.groebner(right_gens, *symbols, order="grevlex", domain=sympy.QQ)
    left_contained = all(_reduces_to_zero(right_basis, expr) for expr in left_gens)
    right_contained = all(_reduces_to_zero(left_basis, expr) for expr in right_gens)
    return bool(left_contained and right_contained)


def _saturation_generators(
    ideal_gens: list[Any], d_expr: Any, symbols: tuple[Any, ...]
) -> list[Any]:
    """Generators of ``<I> : <d>^infinity`` by exact one-step elimination.

    ``QQ[variables][t] / <1 - t*d>`` is the localization at ``d``, so the
    contraction of ``<t*i for i in I> + <1 - t*d>`` to ``QQ[variables]``
    equals ``I : <d>^infinity`` exactly; a lexicographic basis with the
    eliminator first (the greatest lex variable) exposes its generators as
    the eliminator-free elements by the elimination theorem.
    """
    eliminator = sympy.Dummy("jacobian_saturation")
    basis = sympy.groebner(
        [eliminator * expr for expr in ideal_gens] + [1 - eliminator * d_expr],
        eliminator,
        *symbols,
        order="lex",
        domain=sympy.QQ,
    )
    return [
        poly.as_expr() for poly in basis.polys if not poly.as_expr().has(eliminator)
    ]


def verify_saturation_relation(
    ideal: RationalPolynomialIdeal,
    saturation_polynomial: RationalPolynomial,
    claimed: RationalPolynomialIdeal,
) -> None:
    """Replay the defining relation of an exact saturation in-process.

    The exact localization identity ``contraction(<t*I, 1 - t*d>) =
    I : <d>^infinity`` recomputes the saturation with the independent sympy
    kernel inside one bounded elimination; requiring mutual containment with
    the claimed value validates the authoritative derived ideal without
    launching Singular again.  Raises ``ValueError`` when the claim fails.
    """
    symbols = symbols_for_variables(ideal.variables)
    d_expr = rational_polynomial_to_sympy(saturation_polynomial).as_expr()
    recomputed = _saturation_generators(_sympy_generators(ideal), d_expr, symbols)
    if not _mutually_contained(recomputed, _sympy_generators(claimed), symbols):
        raise ValueError(
            "saturation must be the exact saturation of the retained request"
        )


__all__ = [
    "compute_ideal_quotient",
    "compute_ideal_radical",
    "compute_ideal_radical_membership",
    "compute_ideal_saturation",
    "verify_saturation_relation",
]
