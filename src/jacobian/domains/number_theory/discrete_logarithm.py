"""Direct bounded SymPy discrete-logarithm operation."""

from __future__ import annotations

from sympy.ntheory import discrete_log

from jacobian.contracts.number_theory import (
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
)
from jacobian.domains._examples import example
from jacobian.domains.number_theory._support import number_theory_operation


def _compute(request: DiscreteLogarithmRequest) -> DiscreteLogarithmResult:
    try:
        exponent = int(discrete_log(request.modulus, request.target, request.base))
    except ValueError as exc:
        if "Log does not exist" not in str(exc):
            raise
        return DiscreteLogarithmResult(
            status="UNSOLVABLE",
            base=request.base,
            target=request.target,
            modulus=request.modulus,
        )
    return DiscreteLogarithmResult(
        status="SOLVED",
        base=request.base,
        target=request.target,
        modulus=request.modulus,
        discrete_log=exponent,
    )


DISCRETE_LOGARITHM_OPERATION = number_theory_operation(
    "modular.compute.discrete_logarithm",
    "Compute a bounded discrete logarithm",
    "Compute a modular discrete logarithm through the pinned SymPy dependency.",
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
    _compute,
    "number-theory",
    "modular",
    "discrete-logarithm",
    "bounded",
    "sympy",
    version="1",
    examples=(
        example(
            "two_to_one_mod_three",
            "Solve 2^x = 1 modulo 3.",
            {"base": 2, "target": 1, "modulus": 3},
        ),
    ),
)
