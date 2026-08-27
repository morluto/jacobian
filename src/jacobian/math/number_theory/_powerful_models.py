"""Typed contracts for bounded powerful-number decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory._integer_models import PrimePower
from jacobian.math.number_theory._models import _validation_error

if TYPE_CHECKING:
    from jacobian.math.number_theory._powerful_kernels import PowerfulDecisionData

# The kernel derives ``B = ceil(value**(1/5))`` and trial-divides through B.
# These limits bound that one producer pass without requiring complete
# factorization.
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
    """An exact powerful-number decision from one admitted kernel pass."""

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
    def require_branch_consistency(self) -> Self:
        if self.is_powerful != (self.conclusion == "POWERFUL"):
            raise _validation_error(
                "powerful_number_conclusion_boolean_mismatch",
                "is_powerful must agree with conclusion",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PowerfulNumberRequest,
        *,
        data: PowerfulDecisionData,
    ) -> Self:
        """Build one result after the admitted decision kernel established it."""

        from jacobian.canonical import format_canonical_integer

        return cls(
            value=request.value,
            conclusion=data.conclusion,
            is_powerful=data.conclusion == "POWERFUL",
            cutoff=data.cutoff,
            checked_through=data.checked_through,
            stripped_factors=tuple(
                PrimePower(prime=format_canonical_integer(prime), power=exponent)
                for prime, exponent in data.stripped_factors
            ),
            residual=format_canonical_integer(data.residual),
            residual_perfect_power=(
                None
                if data.perfect_power is None
                else ResidualPerfectPower(
                    base=format_canonical_integer(data.perfect_power[0]),
                    exponent=data.perfect_power[1],
                )
            ),
        )


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
