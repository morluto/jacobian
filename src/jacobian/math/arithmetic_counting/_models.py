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
_MAX_FLOOR_SUM_RESULT_DIGITS = 32_768
_MAX_BOX_COORD = 10_000
_MAX_BOX_AREA = 250_000
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
                "parameters, so large n is admitted; |result| <= n*(a+b) stays "
                f"inside the {_MAX_FLOOR_SUM_RESULT_DIGITS}-digit canonical bound."
            )
        }
    )

    n: int = Field(ge=0, le=_MAX_FLOOR_SUM_N)
    m: int = Field(ge=1, le=_MAX_FLOOR_SUM_PARAM)
    a: int = Field(ge=0, le=_MAX_FLOOR_SUM_PARAM)
    b: int = Field(ge=0, le=_MAX_FLOOR_SUM_PARAM)

    @model_validator(mode="after")
    def require_bounded_result(self) -> Self:
        # Worst case m=1: |sum| <= n*(a+b).  Bound its decimal digits before
        # execution so the exact result always fits the canonical contract.
        magnitude_digits = len(str((self.n + 1) * (self.a + self.b) + 1))
        if magnitude_digits > _MAX_FLOOR_SUM_RESULT_DIGITS:
            raise _validation_error(
                "floor_sum_result_exceeds_bound",
                "floor-sum result can exceed the "
                f"{_MAX_FLOOR_SUM_RESULT_DIGITS}-digit canonical integer bound",
            )
        return self


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
            raise _validation_error("x_interval_invalid", "x_lo must be <= x_hi")
        if self.y_lo > self.y_hi:
            raise _validation_error("y_interval_invalid", "y_lo must be <= y_hi")
        area = (self.x_hi - self.x_lo + 1) * (self.y_hi - self.y_lo + 1)
        if area > _MAX_BOX_AREA:
            raise _validation_error(
                "box_area_exceeds_budget", "box area exceeds the computational budget"
            )
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
