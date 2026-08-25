"""Typed contracts and replay validation for powerful-number decisions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._models import PrimePower, _validation_error

# The kernel derives ``B = ceil(value**(1/5))`` and trial-divides through B.
# These limits therefore bound both the request and the source-bound result
# replay without requiring complete factorization.
MAX_POWERFUL_INTEGER_DIGITS = 25
MAX_POWERFUL_CUTOFF = 100_000
MAX_POWERFUL_FACTOR_ENTRIES = 42
MAX_POWERFUL_EXPONENT = 83

PowerfulInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^[1-9][0-9]*$",
        max_length=MAX_POWERFUL_INTEGER_DIGITS,
        strict=True,
    ),
]


class PowerfulNumberRequest(StrictModel):
    """One positive canonical integer of at most 25 digits."""

    value: PowerfulInteger = Field(
        description=(
            "Positive canonical decimal integer with at most 25 digits. The "
            "kernel derives B=ceil(value^(1/5)), so B <= 100000."
        ),
        examples=["12168"],
    )


class ResidualPerfectPower(StrictModel):
    """An exact decomposition of the stripped residual as base**exponent."""

    base: PowerfulInteger
    exponent: StrictInt = Field(ge=2, le=MAX_POWERFUL_EXPONENT)


class PowerfulNumberResult(StrictModel):
    """A source-bound, replayable exact powerful-number decision."""

    value: PowerfulInteger
    conclusion: Literal[
        "POWERFUL",
        "EXPONENT_ONE",
        "ROUGH_NOT_PERFECT_POWER",
    ]
    is_powerful: StrictBool
    cutoff: StrictInt = Field(
        ge=1,
        le=MAX_POWERFUL_CUTOFF,
        description="The canonical cutoff B=ceil(value^(1/5)); B^5 >= value.",
    )
    checked_through: StrictInt = Field(
        ge=1,
        le=MAX_POWERFUL_CUTOFF,
        description=(
            "All primes at most this bound were tested. It equals cutoff unless "
            "an exponent-one prime ended the decision early."
        ),
    )
    stripped_factors: tuple[PrimePower, ...] = Field(
        min_length=0,
        max_length=MAX_POWERFUL_FACTOR_ENTRIES,
    )
    residual: PowerfulInteger = Field(
        description=(
            "The positive cofactor after the reported prime powers are removed."
        )
    )
    residual_perfect_power: ResidualPerfectPower | None = None

    @model_validator(mode="after")
    def bind_decision_to_source_by_exact_replay(self) -> PowerfulNumberResult:
        from jacobian.canonical import parse_canonical_integer
        from jacobian.math.number_theory._powerful_kernels import (
            decide_powerful_data,
        )

        expected = decide_powerful_data(parse_canonical_integer(self.value))
        factors = tuple(
            (parse_canonical_integer(factor.prime), factor.power)
            for factor in self.stripped_factors
        )
        perfect_power = (
            None
            if self.residual_perfect_power is None
            else (
                parse_canonical_integer(self.residual_perfect_power.base),
                self.residual_perfect_power.exponent,
            )
        )
        if (
            self.conclusion != expected.conclusion
            or self.is_powerful != (expected.conclusion == "POWERFUL")
            or self.cutoff != expected.cutoff
            or self.checked_through != expected.checked_through
            or factors != expected.stripped_factors
            or parse_canonical_integer(self.residual) != expected.residual
            or perfect_power != expected.perfect_power
        ):
            raise _validation_error(
                "powerful_number_conclusion_or_certificate_does_not_match_exact_replay",
                "powerful-number conclusion or certificate does not match exact replay",
            )
        return self


__all__ = [
    "MAX_POWERFUL_CUTOFF",
    "MAX_POWERFUL_EXPONENT",
    "MAX_POWERFUL_FACTOR_ENTRIES",
    "MAX_POWERFUL_INTEGER_DIGITS",
    "PowerfulInteger",
    "PowerfulNumberRequest",
    "PowerfulNumberResult",
    "ResidualPerfectPower",
]
