"""Polynomial-derivation operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.derivations._models import (
    DerivationApplyRequest,
    DerivationApplyResult,
    DerivationCertificateRequest,
    DerivationIteratesRequest,
    DerivationIteratesResult,
    GaActionRequest,
    LocallyNilpotentCertificate,
    PolynomialGaAction,
)
from jacobian.math.polynomials.derivations.operations import (
    apply_derivation,
    construct_locally_nilpotent_certificate,
    derivation_iterates,
    ga_action_from_certificate,
)


def _run_apply_derivation(request: DerivationApplyRequest) -> DerivationApplyResult:
    return apply_derivation(request.derivation, request.polynomial)


def _rational(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _poly(variables: list[str], terms: list[tuple[int, list[int]]]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": variables,
        "polynomial": {
            "terms": [
                {"coefficient": _rational(coefficient), "exponents": exponents}
                for coefficient, exponents in terms
            ]
        },
    }


_TRIANGULAR_IMAGES = [
    _poly(["x", "y"], [(1, [0, 1])]),
    _poly(["x", "y"], []),
]

_TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial_derivation.apply.compute",
        title="Apply an exact polynomial derivation to a polynomial",
        description=(
            "Compute D(f) = sum_i D(x_i) * partial_i(f) for a checked "
            "polynomial derivation on QQ[x_1, ..., x_n] with n between 1 and "
            "8 and a source polynomial in the same ordered ring, returning "
            "the canonical result with one contribution polynomial per "
            "generator. Source images and polynomials are bounded in terms, "
            "degree, and coefficient digits, and predicted contribution "
            "cells are preflighted before expansion."
        ),
        request_type=DerivationApplyRequest,
        result_type=DerivationApplyResult,
        run=_run_apply_derivation,
        tags=("polynomial", "derivation", "exact", "ga-action"),
        discovery_terms=(
            "polynomial derivation action",
            "Leibniz rule application",
            "locally nilpotent derivation iterate",
            "additive group infinitesimal action",
        ),
        examples=(
            OperationExample(
                name="triangular_derivation_on_square",
                description=(
                    "Apply D(x)=y, D(y)=0 to x^2 to get 2*x*y with ledger "
                    "(2*x*y, 0); the derivation images and the source must "
                    "share one ordered QQ ring."
                ),
                input={
                    "derivation": {
                        "variables": ["x", "y"],
                        "images": _TRIANGULAR_IMAGES,
                    },
                    "polynomial": _poly(["x", "y"], [(1, [2, 0])]),
                },
            ),
        ),
    ),
)

_ITERATE_EXAMPLE = {
    "derivation": {"variables": ["x", "y"], "images": _TRIANGULAR_IMAGES},
    "polynomial": _poly(["x", "y"], [(1, [2, 0])]),
    "bound": 3,
}
_CERTIFICATE_EXAMPLE = {
    "derivation": {"variables": ["x", "y"], "images": _TRIANGULAR_IMAGES},
    "chains": [
        [
            _poly(["x", "y"], [(1, [1, 0])]),
            _poly(["x", "y"], [(1, [0, 1])]),
            _poly(["x", "y"], []),
        ],
        [_poly(["x", "y"], [(1, [0, 1])]), _poly(["x", "y"], [])],
    ],
}

_TOOLS += (
    MathTool(
        operation_id="polynomial_derivation.iterates.compute",
        title="Compute a bounded exact derivation iterate profile",
        description="Compute D^0(f) through D^N(f), returning the first zero when reached or a nonzero-through-bound profile; a finite nonzero prefix is not a global nonnilpotence claim.",
        request_type=DerivationIteratesRequest,
        result_type=DerivationIteratesResult,
        run=lambda request: derivation_iterates(
            request.derivation, request.polynomial, request.bound
        ),
        tags=("polynomial", "derivation", "iterate", "exact"),
        examples=(
            OperationExample(
                name="triangular_iterates",
                description="Compute the bounded iterates of D(x)=y,D(y)=0 on x^2; the bound is finite and the source ring is shared.",
                input=_ITERATE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="polynomial_derivation.local_nilpotence_certificate.construct",
        title="Construct a checked locally nilpotent derivation certificate",
        description="Check complete exact generator iterate chains ending at zero and return the locally nilpotent derivation certificate; mutated chains are rejected.",
        request_type=DerivationCertificateRequest,
        result_type=LocallyNilpotentCertificate,
        run=lambda request: construct_locally_nilpotent_certificate(
            request.derivation, request.chains
        ),
        tags=("polynomial", "derivation", "locally-nilpotent", "certificate"),
        examples=(
            OperationExample(
                name="triangular_certificate",
                description="Construct the generator-chain certificate for D(x)=y,D(y)=0; every chain must begin at its generator and end at zero.",
                input=_CERTIFICATE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_group.ga.action_from_derivation.compute",
        title="Exponentiate a checked locally nilpotent derivation",
        description="Compute the finite exact additive-group exponential action exp(tD) on generators from a checked locally nilpotent certificate; the parameter is a new polynomial axis.",
        request_type=GaActionRequest,
        result_type=PolynomialGaAction,
        run=lambda request: ga_action_from_certificate(
            construct_locally_nilpotent_certificate(request.derivation, request.chains)
        ),
        tags=("algebraic-group", "ga", "derivation", "exact"),
        examples=(
            OperationExample(
                name="triangular_ga_action",
                description="Exponentiate D(x)=y,D(y)=0 to x maps to x+t*y and y maps to y; the input generator chains must be complete.",
                input=_CERTIFICATE_EXAMPLE,
            ),
        ),
    ),
)

TOOLS: tuple[MathTool[Any, Any], ...] = _TOOLS

__all__ = ["TOOLS"]
