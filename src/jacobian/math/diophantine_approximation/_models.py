"""Typed wire contracts for exact Diophantine approximation operations."""

from __future__ import annotations

from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

_MAX_DISCRIMINANT = 1_000_000
_MAX_TERMS = 5_000


def _is_square_free(value: int) -> bool:
    return all(value % (divisor * divisor) for divisor in range(2, isqrt(value) + 1))


class SquarefreeRequest(StrictModel):
    """One positive squarefree integer D for sqrt(D) operations."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not _is_square_free(self.discriminant):
            raise ValueError("discriminant must be squarefree")
        return self


class ContinuedFractionRequest(StrictModel):
    """Request the continued fraction expansion of sqrt(D) up to n terms."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    term_count: StrictInt = Field(ge=1, le=_MAX_TERMS)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not _is_square_free(self.discriminant):
            raise ValueError("discriminant must be squarefree")
        return self


class ContinuedFractionResult(StrictModel):
    """The continued fraction [a_0; a_1, ...] of sqrt(D)."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    coefficients: tuple[StrictInt, ...] = Field(min_length=1, max_length=_MAX_TERMS)
    preperiod_length: StrictInt = Field(ge=1)
    period_length: StrictInt = Field(ge=1)
    method: Literal["SYMPY_CONTINUED_FRACTION"] = "SYMPY_CONTINUED_FRACTION"


class ConvergentRequest(StrictModel):
    """Request the first n convergents p_n/q_n of sqrt(D)."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    convergent_count: StrictInt = Field(ge=1, le=_MAX_TERMS)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not _is_square_free(self.discriminant):
            raise ValueError("discriminant must be squarefree")
        return self


class ConvergentValue(StrictModel):
    """One convergent p_n/q_n with index n."""

    index: StrictInt = Field(ge=0)
    numerator: CanonicalInteger
    denominator: CanonicalInteger


class ConvergentResult(StrictModel):
    """Convergents of sqrt(D)."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    convergents: tuple[ConvergentValue, ...] = Field(
        min_length=1, max_length=_MAX_TERMS
    )
    method: Literal["CONTINUED_FRACTION_RECURSION"] = "CONTINUED_FRACTION_RECURSION"


class PellEquationRequest(StrictModel):
    """Solve x^2 - D*y^2 = 1 for the fundamental solution."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not _is_square_free(self.discriminant):
            raise ValueError("discriminant must be squarefree")
        return self


class PellEquationResult(StrictModel):
    """The fundamental solution (x, y) to x^2 - D*y^2 = 1."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    x: CanonicalInteger
    y: CanonicalInteger
    method: Literal["CONTINUED_FRACTION_CONVERGENTS"] = "CONTINUED_FRACTION_CONVERGENTS"


__all__ = [
    "ContinuedFractionRequest",
    "ContinuedFractionResult",
    "ConvergentRequest",
    "ConvergentResult",
    "ConvergentValue",
    "PellEquationRequest",
    "PellEquationResult",
    "SquarefreeRequest",
]
