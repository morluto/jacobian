"""Typed declarations for p-adic number theory operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.padic_arithmetic._models import (
    HenselRootRequest,
    HenselRootResult,
    PAdicRootsRequest,
    PAdicRootsResult,
)
from jacobian.math.padic_arithmetic._operations import (
    find_padic_roots,
    hensel_lift_root,
)


def padic_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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


PADIC_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    padic_operation(
        "number_theory.padic.hensel_root.compute",
        "Lift a simple root mod p to a root mod p^k via Hensel's lemma",
        "Given an integer polynomial f, a prime p, and a root r with "
        "f(r) ≡ 0 (mod p) and f'(r) not ≡ 0 (mod p), lift r to a root "
        "mod p^k using Hensel's lemma. Returns the lifted root and whether "
        "the root is simple.",
        HenselRootRequest,
        HenselRootResult,
        hensel_lift_root,
        "p-adic",
        "hensel",
        "root-lifting",
        "exact",
        examples=(
            example(
                "hensel_lift_x_squared_plus_1",
                "Lift the root 2 of x^2+1 mod 5 to a root mod 5^4.",
                _HENSEL_ROOT_EXAMPLE,
            ),
        ),
    ),
    padic_operation(
        "number_theory.padic.roots.compute",
        "Find every simple root of f(x) mod p^k via Hensel lifting",
        "Find every simple root of an integer polynomial f(x) modulo p^k: "
        "residues r mod p with f(r) = 0 and f'(r) != 0 (mod p) lift "
        "uniquely via Hensel's lemma. Residues whose derivative also "
        "vanishes are returned in multiple_residues without lifting, since "
        "their mod-p^k root sets can grow unboundedly.",
        PAdicRootsRequest,
        PAdicRootsResult,
        find_padic_roots,
        "p-adic",
        "root-finding",
        "hensel",
        "exact",
        examples=(
            example(
                "padic_roots_x_cubed_minus_1",
                "Find all roots of x^3-1 mod 5^3.",
                _PADIC_ROOTS_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = PADIC_OPERATIONS

__all__ = ["TOOLS"]
