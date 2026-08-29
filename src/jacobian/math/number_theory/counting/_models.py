"""Typed wire contracts for arithmetic counting operations."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

# The floor-sum kernel is the Euclidean-like recursion (O(log m * log a/m)
# halvings), so ``n`` no longer controls work.  The admitted ceiling stays
# inside the interoperable JSON integer range (< 2^53) so the published
# schema can carry it, and the result preflight below independently bounds
# the exact output digits.
_MAX_FLOOR_SUM_N = 10**15
_MAX_FLOOR_SUM_PARAM = 1_000_000
_MAX_BOX_COORD = 10_000
_MAX_BOX_LINEAR_COEFFICIENT = 10**15
_MAX_BOX_MODULUS = 10_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by arithmetic-counting contracts."""

    return PydanticCustomError(f"arithmetic_counting.{reason}", message)


class FloorSumRequest(StrictModel):
    """Compute sum_{i=0}^{n-1} floor((a*i + b) / m) for bounded non-negative inputs."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Exact sum of floor((a*i+b)/m) for i in [0, n). Executed by the "
                "Euclidean-like recursion whose work is logarithmic in the "
                "parameters, so large n is admitted."
            )
        }
    )

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
    u: int = Field(
        ge=-_MAX_BOX_LINEAR_COEFFICIENT,
        le=_MAX_BOX_LINEAR_COEFFICIENT,
    )
    v: int = Field(
        ge=-_MAX_BOX_LINEAR_COEFFICIENT,
        le=_MAX_BOX_LINEAR_COEFFICIENT,
    )
    c: int = Field(
        ge=-_MAX_BOX_LINEAR_COEFFICIENT,
        le=_MAX_BOX_LINEAR_COEFFICIENT,
    )
    modulus: int = Field(ge=1, le=_MAX_BOX_MODULUS)

    @model_validator(mode="after")
    def require_valid_box(self) -> Self:
        if self.x_lo > self.x_hi:
            raise _validation_error("x_interval_invalid", "x_lo must be <= x_hi")
        if self.y_lo > self.y_hi:
            raise _validation_error("y_interval_invalid", "y_lo must be <= y_hi")
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
