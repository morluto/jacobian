"""Structured exact differential and residue values over QQ(x)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import RationalFunction, RationalPolynomial


class RationalFunctionRequest(StrictModel):
    function: RationalFunction


class LogarithmicDifferentialTerm(StrictModel):
    """A rationally represented dlog row; no analytic log branch is chosen."""

    factor: RationalPolynomial
    numerator: RationalPolynomial
    exponent: Literal[1] = 1


class LogarithmicDifferentialResult(StrictModel):
    source: RationalFunction
    terms: tuple[LogarithmicDifferentialTerm, ...] = Field(default=())
    reconstructed: RationalFunction


class ResidueRow(StrictModel):
    factor: RationalPolynomial
    root_index: int = Field(ge=0)
    pole_order: int = Field(ge=1)
    numerator_at_root: RationalPolynomial
    derivative_factor: RationalPolynomial


class GlobalResidueResult(StrictModel):
    source: RationalFunction
    finite_poles: tuple[ResidueRow, ...]
    residue_at_infinity: CanonicalRational
    finite_residue_sum: CanonicalRational
    total_residue: CanonicalRational


class FormalAntiderivativeResult(StrictModel):
    source: RationalFunction
    rational_part: RationalFunction
    logarithmic_part: LogarithmicDifferentialResult


class RationalPrimitiveResult(StrictModel):
    source: RationalFunction
    status: Literal["RATIONAL_PRIMITIVE", "NO_RATIONAL_PRIMITIVE"]
    rational_part: RationalFunction
    remainder: RationalFunction

    @model_validator(mode="after")
    def require_status_remainder(self) -> Self:
        has_remainder = bool(self.remainder.numerator.terms)
        if self.status == "RATIONAL_PRIMITIVE" and has_remainder:
            raise ValueError(
                "a rational-primitive result must carry a zero Hermite remainder"
            )
        if self.status == "NO_RATIONAL_PRIMITIVE" and not has_remainder:
            raise ValueError(
                "a non-primitive result must carry a nonzero Hermite remainder"
            )
        return self


__all__ = [
    "FormalAntiderivativeResult",
    "GlobalResidueResult",
    "LogarithmicDifferentialResult",
    "LogarithmicDifferentialTerm",
    "RationalFunctionRequest",
    "RationalPrimitiveResult",
    "ResidueRow",
]
