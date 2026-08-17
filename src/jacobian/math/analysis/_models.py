"""Typed contracts for rigorous real-function point enclosures."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_RATIONAL_DIGITS = 128


class RealUnaryFunction(StrEnum):
    EXP = "EXP"
    LOG = "LOG"
    SQRT = "SQRT"
    SIN = "SIN"
    COS = "COS"


class ArbPointEnclosureRequest(StrictModel):
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(default=128, ge=32, le=4096)

    @model_validator(mode="after")
    def bound_argument_size(self) -> Self:
        require_bounded_rational(
            self.argument,
            max_digits=MAX_RATIONAL_DIGITS,
            label="validated-analysis rational",
        )
        return self


class ExactDyadic(StrictModel):
    """The exact value ``mantissa * 2**exponent``."""

    mantissa: str = Field(pattern=r"^-?(?:0|[1-9][0-9]*)$")
    exponent: StrictInt

    @model_validator(mode="after")
    def require_canonical_binary_form(self) -> Self:
        mantissa = int(self.mantissa)
        if mantissa == 0 and self.exponent != 0:
            raise ValueError("canonical dyadic zero must have exponent 0")
        if mantissa != 0 and mantissa % 2 == 0:
            raise ValueError("canonical nonzero dyadic mantissa must be odd")
        return self

    def as_fraction(self) -> Fraction:
        mantissa = Fraction(int(self.mantissa))
        if self.exponent >= 0:
            return mantissa * Fraction(2**self.exponent, 1)
        return mantissa / Fraction(2 ** (-self.exponent), 1)


class ArbPointEnclosureResult(StrictModel):
    status: Literal["ENCLOSED", "NONFINITE", "TIMEOUT", "BACKEND_ERROR"]
    function: RealUnaryFunction
    argument: CanonicalRational
    precision_bits: StrictInt = Field(ge=32, le=4096)
    lower: ExactDyadic | None = None
    upper: ExactDyadic | None = None
    relative_accuracy_bits: StrictInt | None = None
    exact: bool = False
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_enclosure_to_status(self) -> Self:
        enclosed = self.status == "ENCLOSED"
        if enclosed != (self.lower is not None and self.upper is not None):
            raise ValueError("only an enclosed result may carry dyadic endpoints")
        if not enclosed and (self.relative_accuracy_bits is not None or self.exact):
            raise ValueError("a non-enclosure cannot claim accuracy or exactness")
        if enclosed:
            lower = self.lower
            upper = self.upper
            if lower is None or upper is None:
                raise ValueError("only an enclosed result may carry dyadic endpoints")
            if lower.as_fraction() > upper.as_fraction():
                raise ValueError("enclosure lower endpoint exceeds upper endpoint")
            if self.exact != (self.relative_accuracy_bits is None):
                raise ValueError(
                    "exact enclosures omit relative accuracy; inexact ones report it"
                )
        return self
