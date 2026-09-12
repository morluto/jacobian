"""Typed wire contracts for arithmetic counting operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding, ExactInteger
from jacobian._models import StrictModel

# The floor-sum kernel is the Euclidean-like recursion (O(log m * log a/m)
# halvings), so ``n`` no longer controls work.  The admitted ceiling stays
# inside the interoperable JSON integer range (< 2^53) so the published
# schema can carry it, and the result preflight below independently bounds
# the exact output digits.
_MAX_FLOOR_SUM_N = 10**15
_MAX_FLOOR_SUM_PARAM = 1_000_000
# The residue aggregation kernel visits one class modulo ``modulus``, not one
# lattice point or coordinate value.  Thirty-two decimal digits leave a
# compact, explicit bound for both the endpoint arithmetic and the exact
# count: a box has fewer than 4 * 10**64 points.
_MAX_BOX_COORD_DIGITS = 32
_MAX_BOX_LINEAR_COEFFICIENT = 10**15
_MAX_BOX_MODULUS = 10_000

BoxCoordinate = Annotated[int, DecimalIntegerEncoding(max_digits=_MAX_BOX_COORD_DIGITS)]
_BOX_COORDINATE_DESCRIPTION = (
    "Exact signed coordinate with at most 32 decimal digits; use a canonical "
    "decimal string in JSON."
)


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

    n: int = Field(ge=0, le=_MAX_FLOOR_SUM_N)
    m: int = Field(ge=1, le=_MAX_FLOOR_SUM_PARAM)
    a: int = Field(ge=0, le=_MAX_FLOOR_SUM_PARAM)
    b: int = Field(ge=0, le=_MAX_FLOOR_SUM_PARAM)
    value: ExactInteger


class CongruenceBoxCountRequest(StrictModel):
    """Count lattice points in a box satisfying a linear congruence."""

    x_lo: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    x_hi: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    y_lo: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    y_hi: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
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

    x_lo: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    x_hi: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    y_lo: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    y_hi: BoxCoordinate = Field(description=_BOX_COORDINATE_DESCRIPTION)
    u: int = Field(ge=-_MAX_BOX_LINEAR_COEFFICIENT, le=_MAX_BOX_LINEAR_COEFFICIENT)
    v: int = Field(ge=-_MAX_BOX_LINEAR_COEFFICIENT, le=_MAX_BOX_LINEAR_COEFFICIENT)
    c: int = Field(ge=-_MAX_BOX_LINEAR_COEFFICIENT, le=_MAX_BOX_LINEAR_COEFFICIENT)
    count: ExactInteger = Field(ge=0)
    modulus: int = Field(ge=1, le=_MAX_BOX_MODULUS)


__all__ = [
    "CongruenceBoxCountRequest",
    "CongruenceBoxCountResult",
    "FloorSumRequest",
    "FloorSumResult",
]
