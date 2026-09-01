"""Typed declarations for p-adic number theory operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.p_adic._models import (
    HenselRootRequest,
    HenselRootResult,
    PAdicRootsRequest,
    PAdicRootsResult,
)
from jacobian.math.number_theory.p_adic.operations import (
    find_padic_roots,
    hensel_lift_root,
)


def _hensel_lift_root(request: HenselRootRequest) -> HenselRootResult:
    return hensel_lift_root(
        request.polynomial, request.prime, request.root_mod_p, request.precision
    )


def _find_padic_roots(request: PAdicRootsRequest) -> PAdicRootsResult:
    return find_padic_roots(request.polynomial, request.prime, request.precision)


_HENSEL_ROOT_EXAMPLE: dict[str, Any] = {
    "polynomial": {"coefficients": ["1", "0", "1"]},
    "prime": 5,
    "root_mod_p": 2,
    "precision": 4,
}

_PADIC_ROOTS_EXAMPLE: dict[str, Any] = {
    "polynomial": {"coefficients": ["-1", "0", "0", "1"]},
    "prime": 5,
    "precision": 3,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="number_theory.padic.hensel_root.compute",
        title="Lift a simple root mod p to a root mod p^k via Hensel's lemma",
        description="Given an integer polynomial f, a prime p, and a root r with "
        "f(r) ≡ 0 (mod p) and f'(r) not ≡ 0 (mod p), lift r to a root "
        "mod p^k using Hensel's lemma. Returns the lifted root and whether "
        "the root is simple.",
        request_type=HenselRootRequest,
        result_type=HenselRootResult,
        run=_hensel_lift_root,
        tags=("p-adic", "hensel", "root-lifting", "exact"),
        examples=(
            OperationExample(
                name="hensel_lift_x_squared_plus_1",
                description="Lift the root 2 of x^2+1 mod 5 to a root mod 5^4.",
                input=_HENSEL_ROOT_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.padic.roots.compute",
        title="Find every simple root of f(x) mod p^k via Hensel lifting",
        description="Find every simple root of an integer polynomial f(x) modulo p^k: "
        "residues r mod p with f(r) = 0 and f'(r) != 0 (mod p) lift "
        "uniquely via Hensel's lemma. Residues whose derivative also "
        "vanishes are returned in multiple_residues without lifting, since "
        "their mod-p^k root sets can grow unboundedly.",
        request_type=PAdicRootsRequest,
        result_type=PAdicRootsResult,
        run=_find_padic_roots,
        tags=("p-adic", "root-finding", "hensel", "exact"),
        examples=(
            OperationExample(
                name="padic_roots_x_cubed_minus_1",
                description="Find all roots of x^3-1 mod 5^3.",
                input=_PADIC_ROOTS_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
