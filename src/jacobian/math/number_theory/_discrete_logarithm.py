"""Direct bounded discrete-logarithm operation."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.math.number_theory._models import _validation_error
from jacobian.math.number_theory._modular_basic_models import MAX_MODULUS
from jacobian.math.number_theory._support import number_theory_operation


class DiscreteLogarithmRequest(StrictModel):
    """One bounded modular discrete-logarithm problem."""

    base: StrictInt = Field(ge=0, le=MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=MAX_MODULUS)

    @model_validator(mode="after")
    def require_canonical_residues(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise _validation_error(
                "base_and_target_must_be_less_than_the_modulus",
                "base and target must be less than the modulus",
            )
        return self


class DiscreteLogarithmResult(StrictModel):
    """The exact result of one bounded discrete-logarithm computation."""

    status: Literal["SOLVED", "UNSOLVABLE"]
    base: StrictInt = Field(ge=0, le=MAX_MODULUS)
    target: StrictInt = Field(ge=0, le=MAX_MODULUS)
    modulus: StrictInt = Field(ge=2, le=MAX_MODULUS)
    discrete_log: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def bind_conclusion(self) -> Self:
        if self.base >= self.modulus or self.target >= self.modulus:
            raise _validation_error(
                "base_and_target_must_be_less_than_the_modulus",
                "base and target must be less than the modulus",
            )
        if self.status == "SOLVED":
            if self.discrete_log is None:
                raise _validation_error(
                    "solved_discrete_logarithm_requires_an_exponent",
                    "solved discrete logarithm requires an exponent",
                )
            if pow(self.base, self.discrete_log, self.modulus) != self.target:
                raise _validation_error(
                    "discrete_logarithm_does_not_reproduce_the_target",
                    "discrete logarithm does not reproduce the target",
                )
        elif self.discrete_log is not None:
            raise _validation_error(
                "unsolvable_discrete_logarithm_cannot_carry_an_exponent",
                "unsolvable discrete logarithm cannot carry an exponent",
            )
        return self


def _compute(request: DiscreteLogarithmRequest) -> DiscreteLogarithmResult:
    """Solve base^x ≡ target (mod modulus) by brute-force search.

    Unlike the group-theoretic SymPy ``discrete_log``, this works for any
    modular equation, including cases where base is not a unit modulo modulus.
    """
    modulus = request.modulus
    base = request.base % modulus
    target = request.target % modulus
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
