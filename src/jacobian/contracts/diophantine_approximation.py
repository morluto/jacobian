"""Typed wire contracts for exact Diophantine approximation operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class SquarefreeRequest(ContractModel):
    """One positive squarefree integer D for sqrt(D) operations."""

    discriminant: int = Field(ge=2, le=10000)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        from sympy import factorint

        if all(e == 1 for e in factorint(self.discriminant).values()):
            return self
        raise ValueError("discriminant must be squarefree")


class ContinuedFractionRequest(ContractModel):
    """Request the continued fraction expansion of sqrt(D) up to n terms."""

    discriminant: int = Field(ge=2, le=10000)
    term_count: int = Field(ge=1, le=500)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        from sympy import factorint

        if all(e == 1 for e in factorint(self.discriminant).values()):
            return self
        raise ValueError("discriminant must be squarefree")


class ContinuedFractionResult(ContractModel):
    """The continued fraction [a_0; a_1, ...] of sqrt(D)."""

    discriminant: int = Field(ge=2, le=10000)
    coefficients: tuple[int, ...] = Field(min_length=1, max_length=500)
    preperiod_length: int = Field(ge=1)
    period_length: int = Field(ge=1)
    method: Literal["SYMPY_CONTINUED_FRACTION"] = "SYMPY_CONTINUED_FRACTION"


class ConvergentRequest(ContractModel):
    """Request the first n convergents p_n/q_n of sqrt(D)."""

    discriminant: int = Field(ge=2, le=10000)
    convergent_count: int = Field(ge=1, le=500)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        from sympy import factorint

        if all(e == 1 for e in factorint(self.discriminant).values()):
            return self
        raise ValueError("discriminant must be squarefree")


class ConvergentValue(ContractModel):
    """One convergent p_n/q_n with index n."""

    index: int = Field(ge=0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=1)


class ConvergentResult(ContractModel):
    """Convergents of sqrt(D)."""

    discriminant: int = Field(ge=2, le=10000)
    convergents: tuple[ConvergentValue, ...] = Field(min_length=1, max_length=500)
    method: Literal["CONTINUED_FRACTION_RECURSION"] = "CONTINUED_FRACTION_RECURSION"


class PellEquationRequest(ContractModel):
    """Solve x^2 - D*y^2 = 1 for the fundamental solution."""

    discriminant: int = Field(ge=2, le=10000)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        from sympy import factorint

        if all(e == 1 for e in factorint(self.discriminant).values()):
            return self
        raise ValueError("discriminant must be squarefree")


class PellEquationResult(ContractModel):
    """The fundamental solution (x, y) to x^2 - D*y^2 = 1."""

    discriminant: int = Field(ge=2, le=10000)
    x: int = Field(ge=1)
    y: int = Field(ge=1)
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
