"""Typed public operation for exact Ramanujan sums."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.math.number_theory._models import _MAX_INTEGER_LENGTH
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.ramanujan_sums import ramanujan_sum

# This matches the established in-process factorization envelope.  SymPy
# factors the modulus once; the frequency only participates in bounded modular
# reductions.  The exact result has absolute value at most the modulus.
_MAX_MODULUS_DIGITS = 12

RamanujanModulus = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]*)$",
        max_length=_MAX_MODULUS_DIGITS,
        strict=True,
    ),
]

# The frequency and exact sum bind as canonical signed decimal integers
# under the same grammar Jacobian's canonical integer encoding uses
# everywhere else: zero is exactly "0" and any other value carries an
# optional minus sign on a nonzero leading digit.  Unlike
# ``BoundedInteger``, whose grammar also admits "-0", this pattern cannot
# bind negative zero, so every accepted string is the unique canonical
# decimal form of its integer and mathematically equal inputs share one
# serialized identity across request and result.
RamanujanSumInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        max_length=_MAX_INTEGER_LENGTH,
        strict=True,
    ),
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
    def bind_value_to_source(self) -> Self:
        expected = ramanujan_sum(int(self.modulus), int(self.frequency))
        if int(self.value) != expected:
            raise ValueError("Ramanujan-sum value does not match its source")
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
    version="1",
)
