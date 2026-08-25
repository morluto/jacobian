"""Typed public operation for exact Ramanujan sums."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.math.number_theory._models import _MAX_INTEGER_LENGTH
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.ramanujan_sums import (
    _MAX_MODULUS_DIGITS,
    ramanujan_sum,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable semantic error owned by the number-theory domain."""

    return PydanticCustomError(f"number_theory.{reason}", message)


RamanujanModulus = Annotated[
    CanonicalInteger,
    Field(max_length=_MAX_MODULUS_DIGITS),
]

# Every field binds through ``jacobian._exact.CanonicalInteger``, the owner of
# Jacobian's canonical signed-decimal integer encoding: zero is exactly "0" and
# every other value carries an optional minus sign on a nonzero leading digit,
# so negative zero cannot bind and mathematically equal inputs share one
# serialized identity across request and result.  Operation-specific execution
# bounds are applied on top of the owner type, never by restating its grammar;
# the modulus's nonnegativity is a mathematical precondition enforced by model
# validation rather than part of the owned encoding.
RamanujanSumInteger = Annotated[
    CanonicalInteger,
    Field(max_length=_MAX_INTEGER_LENGTH),
]


class RamanujanSumRequest(StrictModel):
    """One bounded modulus and signed frequency for an exact Ramanujan sum."""

    modulus: RamanujanModulus = Field(
        description=(
            "Canonical nonnegative integer modulus with at most 12 digits; "
            "zero denotes the empty reduced-residue sum."
        )
    )
    frequency: RamanujanSumInteger = Field(
        description=(
            "Canonical signed integer frequency with at most 256 characters; "
            'zero is exactly "0" and every other value carries an optional '
            "minus sign with no leading zeros."
        )
    )

    @model_validator(mode="after")
    def require_nonnegative_modulus(self) -> Self:
        if int(self.modulus) < 0:
            raise _validation_error(
                "modulus_must_be_nonnegative", "modulus must be nonnegative"
            )
        return self


class RamanujanSumResult(StrictModel):
    """An exact Ramanujan sum bound to its modulus and frequency."""

    modulus: RamanujanModulus
    frequency: RamanujanSumInteger
    value: RamanujanSumInteger = Field(
        description=(
            "Exact Ramanujan sum as a canonical signed decimal integer: "
            'zero is exactly "0" and every other value carries an optional '
            "minus sign with no leading zeros."
        )
    )

    @model_validator(mode="after")
    def require_nonnegative_modulus(self) -> Self:
        if int(self.modulus) < 0:
            raise _validation_error(
                "modulus_must_be_nonnegative", "modulus must be nonnegative"
            )
        return self

    @model_validator(mode="after")
    def bind_value_to_source(self) -> Self:
        expected = ramanujan_sum(int(self.modulus), int(self.frequency))
        if int(self.value) != expected:
            raise _validation_error(
                "ramanujan_sum_value_does_not_match_its_source",
                "Ramanujan-sum value does not match its source",
            )
        return self


def compute_ramanujan_sum(request: RamanujanSumRequest) -> RamanujanSumResult:
    """Evaluate one admitted exact Ramanujan sum."""

    value = ramanujan_sum(int(request.modulus), int(request.frequency))
    return RamanujanSumResult(
        modulus=request.modulus,
        frequency=request.frequency,
        value=str(value),
    )


RAMANUJAN_SUM_OPERATION = number_theory_operation(
    "number_theory.ramanujan_sum.compute",
    "Compute an exact Ramanujan sum",
    (
        "Compute c_q(n) exactly for one bounded nonnegative modulus and signed "
        "frequency using the divisor-Mobius identity through a prime-power "
        "factorization, with the convention c_0(n)=0."
    ),
    RamanujanSumRequest,
    RamanujanSumResult,
    compute_ramanujan_sum,
    "number-theory",
    "ramanujan-sum",
    "mobius",
    "exact",
    examples=(
        example(
            "ramanujan_sum_4_at_2",
            (
                "Compute c_4(2)=-2 exactly; the modulus must be a canonical "
                "nonnegative integer of at most 12 digits and the frequency "
                "must be a canonical signed integer."
            ),
            {"modulus": "4", "frequency": "2"},
        ),
    ),
)
