"""Typed contracts for arithmetic counting operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel


class FloorSumRequest(ContractModel):
    """Request to compute sum_{i=0}^{n-1} floor((a*i+b)/m)."""

    n: StrictInt = Field(ge=0, le=100000)
    m: StrictInt = Field(gt=0, le=100000)
    a: StrictInt = Field(ge=0, le=100000)
    b: StrictInt = Field(ge=0, le=100000)


class FloorSumResult(ContractModel):
    """Result of a floor sum computation."""

    value: StrictInt = Field(ge=0)


class CongruenceConstrainedCountRequest(ContractModel):
    """Count lattice points (x,y) with lower <= x <= upper, 1 <= y <= m, and y ≡ c*x (mod k)."""

    k: StrictInt = Field(gt=0, le=10000, description="Modulus k > 0.")
    m: StrictInt = Field(gt=0, le=10000, description="Upper bound on both coordinates.")
    n: StrictInt = Field(ge=0, le=10000, description="Multiplier for congruence y ≡ n*x mod k.")
    lower: StrictInt = Field(ge=0, le=10000, description="Lower bound on b1 (inclusive).")
    upper: StrictInt = Field(ge=0, le=10000, description="Upper bound on b1 (inclusive).")

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("lower must be <= upper")
        return self


class CongruenceConstrainedCountResult(ContractModel):
    """Result of a congruence-constrained lattice point count."""

    count: StrictInt = Field(ge=0)
    detail: str = Field(min_length=1, max_length=1024)
