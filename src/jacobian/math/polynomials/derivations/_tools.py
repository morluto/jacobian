"""Polynomial-derivation operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.derivations._models import (
    DerivationApplyRequest,
    DerivationApplyResult,
)
from jacobian.math.polynomials.derivations.operations import apply_derivation


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

TOOLS: tuple[MathTool[Any, Any], ...] = _TOOLS

__all__ = ["TOOLS"]
