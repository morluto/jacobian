"""Typed wire contracts for inverse multiplicative function operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel

MAX_N = 100000
MAX_POWER_SUM_EXPONENT = 20


class EulerPhiPreimageRequest(StrictModel):
    """Compute the preimage of the Euler totient function."""

    target: int = Field(ge=1, le=MAX_N)


class EulerPhiPreimageCountRequest(StrictModel):
    """Count the preimage of the Euler totient function."""

    target: int = Field(ge=1, le=MAX_N)


class EulerPhiPowerSumRequest(StrictModel):
    """Compute sum of k-th powers of preimage of phi."""

    target: int = Field(ge=1, le=MAX_N)
    exponent: int = Field(ge=1, le=MAX_POWER_SUM_EXPONENT)


# Results


class EulerPhiPreimageResult(StrictModel):
    target: ExactInteger = Field(ge=1, le=MAX_N)
    preimage: tuple[ExactInteger, ...]
    count: ExactInteger = Field(ge=0)


class EulerPhiPreimageCountResult(StrictModel):
    target: ExactInteger = Field(ge=1, le=MAX_N)
    count: ExactInteger = Field(ge=0)


class EulerPhiPowerSumResult(StrictModel):
    target: ExactInteger = Field(ge=1, le=MAX_N)
    exponent: ExactInteger = Field(ge=1, le=MAX_POWER_SUM_EXPONENT)
    power_sum: ExactInteger = Field(ge=0)
    count: ExactInteger = Field(ge=0)
