"""Typed wire contracts for arithmetic counting operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

_MAX_FLOOR_SUM_N = 1_000_000
_MAX_FLOOR_SUM_PARAM = 1_000_000
_MAX_BOX_COORD = 10_000
_MAX_BOX_AREA = 250_000
_MAX_BOX_MODULUS = 10_000


class FloorSumRequest(StrictModel):
    """Compute sum_{i=0}^{n-1} floor((a*i + b) / m) for bounded non-negative inputs."""

    n: int = Field(ge=0, le=_MAX_FLOOR_SUM_N)
    m: int = Field(ge=1, le=_MAX_FLOOR_SUM_PARAM)
    a: int = Field(ge=0, le=_MAX_FLOOR_SUM_PARAM)
    b: int = Field(ge=0, le=_MAX_FLOOR_SUM_PARAM)


class FloorSumResult(StrictModel):
    """The exact floor sum value."""

    value: CanonicalInteger


class CongruenceBoxCountRequest(StrictModel):
    """Count lattice points in a box satisfying a linear congruence."""

    x_lo: int = Field(ge=-_MAX_BOX_COORD, le=_MAX_BOX_COORD)
    x_hi: int = Field(ge=-_MAX_BOX_COORD, le=_MAX_BOX_COORD)
    y_lo: int = Field(ge=-_MAX_BOX_COORD, le=_MAX_BOX_COORD)
    y_hi: int = Field(ge=-_MAX_BOX_COORD, le=_MAX_BOX_COORD)
    u: int
    v: int
    c: int
    modulus: int = Field(ge=1, le=_MAX_BOX_MODULUS)

    @model_validator(mode="after")
    def require_valid_box(self) -> Self:
        if self.x_lo > self.x_hi:
            raise ValueError("x_lo must be <= x_hi")
        if self.y_lo > self.y_hi:
            raise ValueError("y_lo must be <= y_hi")
        area = (self.x_hi - self.x_lo + 1) * (self.y_hi - self.y_lo + 1)
        if area > _MAX_BOX_AREA:
            raise ValueError("box area exceeds the computational budget")
        return self


class CongruenceBoxCountResult(StrictModel):
    """Count and residue-class ledger."""

    count: int = Field(ge=0)
    modulus: int = Field(gt=0)


__all__ = [
    "CongruenceBoxCountRequest",
    "CongruenceBoxCountResult",
    "FloorSumRequest",
    "FloorSumResult",
]
