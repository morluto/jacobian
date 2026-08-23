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
from jacobian.math.polynomials.values import RationalPolynomialIdeal


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
        t * rational_polynomial_to_sympy(request.denominator).as_expr() - 1,
    ]
    elimination = sympy.groebner(polys, t, *variables, order="lex")
    # Basis elements free of t generate the elimination ideal I : <d>^infinity.
    # Absent any such element the intersection with QQ[vars] is the ZERO
    # ideal — e.g. (0) : <d>^infinity = (0) — not the whole ring; a whole-ring
    # saturation instead shows up as a constant basis element.
    saturated = [expr for expr in elimination.exprs if not expr.has(t) and expr != 0]
    return _groebner_signature(variables, saturated)


def compute_ideal_saturation(request: IdealSaturationRequest) -> IdealSaturationResult:
    """Compute I : <d>^infinity through the bounded Singular backend."""

    denominator = RationalPolynomialIdeal(
        variables=request.denominator.variables,
        generators=(request.denominator,),
    )
    backend = run_singular_ideal_operation(
        "saturation",
        request.ideal,
        denominator,
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


def _saturation_replay_worker(payload: dict, queue) -> None:
    """Child-process body of the bounded saturation replay."""
    try:
        from jacobian.math.commutative_algebra_ops._models import (
            IdealSaturationRequest,
        )

        request = IdealSaturationRequest.model_validate(payload)
        queue.put(("ok", replay_saturation(request)))
    except Exception as exc:  # noqa: BLE001 - reported to the parent verbatim
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def replay_saturation_bounded(request) -> tuple:
    """Run the elimination replay in a killable child process.

    The host-side Groebner computation gets the same wall-clock bound the
    Singular subprocess gets (resource_budget.wall_seconds): an expensive
    replay is terminated and surfaces as a typed validation failure instead
    of hanging result construction.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_saturation_replay_worker,
        args=(request.model_dump(mode="json"), queue),
    )
    process.start()
    process.join(timeout=float(request.resource_budget.wall_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise ValueError(
            "saturation source replay exceeded its "
            f"{request.resource_budget.wall_seconds}-second bounded budget"
        )
    if process.exitcode != 0:
        raise ValueError("saturation source replay worker failed")
    try:
        kind, payload = queue.get(timeout=10)
    except Exception as exc:
        raise ValueError("saturation source replay produced no verdict") from exc
    if kind != "ok":
        raise ValueError(f"saturation source replay failed: {payload}")
    return payload
