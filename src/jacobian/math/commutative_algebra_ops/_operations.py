"""Domain functions for commutative algebra operations."""

from __future__ import annotations

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
    run_singular_saturation_verification,
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

    import time

    from jacobian.math.polynomials.values import (
        RationalPolynomialIdeal,
    )

    saturation_ideal = RationalPolynomialIdeal(
        variables=request.ideal.variables,
        generators=(request.saturation_polynomial,),
    )
    started = time.monotonic()
    backend = run_singular_ideal_operation(
        "saturation",
        request.ideal,
        saturation_ideal,
        request.resource_budget,
    )
    if backend.outcome == "COMPUTED" and backend.ideal is not None:
        # Defining-equality verification (I : d^inf == J) runs inside the
        # same bounded, supervised Singular subprocess flow, never as an
        # unbounded host-process Groebner computation. The declared wall
        # budget covers the complete operation: verification receives only
        # the time the first computation left unspent.
        elapsed = time.monotonic() - started
        remaining = request.resource_budget.wall_seconds - int(elapsed + 0.5)
        if remaining < 1:
            return IdealSaturationResult(
                outcome="TIMEOUT",
                source_ideal=request.ideal,
                source_polynomial=request.saturation_polynomial,
                detail="The resource budget was exhausted by the saturation "
                "computation before its bounded verification could run.",
            )
        verification_budget = request.resource_budget.model_copy(
            update={"wall_seconds": remaining}
        )
        verdict = run_singular_saturation_verification(
            request.ideal,
            saturation_ideal,
            backend.ideal,
            verification_budget,
        )
        if verdict == "REFUTED":
            # A refuted verification means the backend's value cannot be
            # reported; the accepted request still deserves a typed outcome
            # rather than a host exception at the transport boundary.
            return IdealSaturationResult(
                outcome="ERROR",
                source_ideal=request.ideal,
                source_polynomial=request.saturation_polynomial,
                detail="Singular returned an ideal that differs from the exact "
                "saturation I : d^infinity; the computed value was discarded.",
            )
        if verdict != "VERIFIED":
            detail = {
                "UNAVAILABLE": "The supported Singular 4.4 backend is not "
                "installed for saturation verification.",
                "TIMEOUT": "Saturation verification exceeded the declared "
                "wall-time limit.",
                "ERROR": "Singular could not decide the saturation's "
                "defining relation.",
            }.get(verdict, "Saturation verification failed.")
            return IdealSaturationResult(
                outcome=verdict,
                source_ideal=request.ideal,
                source_polynomial=request.saturation_polynomial,
                detail=detail,
            )
    return IdealSaturationResult(
        outcome=backend.outcome,
        source_ideal=request.ideal,
        source_polynomial=request.saturation_polynomial,
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
