"""Typed wire contracts for arithmetic counting operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger


class FloorSumRequest(ContractModel):
    """Compute sum_{i=0}^{n-1} floor((a*i + b) / m)."""

    n: CanonicalInteger
    m: CanonicalInteger
    a: CanonicalInteger
    b: CanonicalInteger


class FloorSumResult(ContractModel):
    """The exact floor sum value."""

    value: CanonicalInteger


class CongruenceBoxCountRequest(ContractModel):
    """Count lattice points in a box satisfying a linear congruence."""

    x_lo: int = Field(ge=-10_000_000, le=10_000_000)
    x_hi: int = Field(ge=-10_000_000, le=10_000_000)
    y_lo: int = Field(ge=-10_000_000, le=10_000_000)
    y_hi: int = Field(ge=-10_000_000, le=10_000_000)
    u: int
    v: int
    c: int
    modulus: int = Field(gt=0)

    @model_validator(mode="after")
    def require_valid_box(self) -> Self:
        if self.x_lo > self.x_hi:
            raise ValueError("x_lo must be <= x_hi")
        if self.y_lo > self.y_hi:
            raise ValueError("y_lo must be <= y_hi")
        return self


class CongruenceBoxCountResult(ContractModel):
    """Count and residue-class ledger."""

    count: int = Field(ge=0)
    modulus: int = Field(gt=0)


__all__ = [
    "CongruenceBoxCountRequest",
    "CongruenceBoxCountResult",
    "FloorSumRequest",
    "FloorSumResult",
]
