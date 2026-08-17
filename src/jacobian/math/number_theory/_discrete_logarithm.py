"""Direct bounded discrete-logarithm operation."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.math.number_theory._models import (
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
)
from jacobian.math.number_theory._support import number_theory_operation


def _compute(request: DiscreteLogarithmRequest) -> DiscreteLogarithmResult:
    """Solve base^x ≡ target (mod modulus) by brute-force search.

    Unlike the group-theoretic SymPy ``discrete_log``, this works for any
    modular equation, including cases where base is not a unit modulo modulus.
    """
    modulus = request.modulus
    base = request.base % modulus
    target = request.target % modulus
    if base == 0 and target != 0:
        return DiscreteLogarithmResult(
            status="UNSOLVABLE",
            base=request.base,
            target=request.target,
            modulus=modulus,
        )
    value = 1 % modulus
    for exponent in range(modulus):
        if value == target:
            return DiscreteLogarithmResult(
                status="SOLVED",
                base=request.base,
                target=request.target,
                modulus=modulus,
                discrete_log=exponent,
            )
        value = (value * base) % modulus
    return DiscreteLogarithmResult(
        status="UNSOLVABLE",
        base=request.base,
        target=request.target,
        modulus=modulus,
    )


DISCRETE_LOGARITHM_OPERATION = number_theory_operation(
    "modular.compute.discrete_logarithm",
    "Compute a bounded discrete logarithm",
    "Compute a modular discrete logarithm through bounded brute-force search.",
    DiscreteLogarithmRequest,
    DiscreteLogarithmResult,
    _compute,
    "number-theory",
    "modular",
    "discrete-logarithm",
    "bounded",
    "brute-force",
    version="1",
    examples=(
        example(
            "two_to_one_mod_three",
            "Solve 2^x = 1 modulo 3.",
            {"base": 2, "target": 1, "modulus": 3},
        ),
        example(
            "three_to_two_mod_five",
            "Solve 3^x = 2 modulo 5; base and target must each be less than the modulus.",
            {"base": 3, "target": 2, "modulus": 5},
        ),
    ),
)
