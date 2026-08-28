"""Typed wire contracts for inverse multiplicative function operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel

MAX_N = 100000


class EulerPhiPreimageRequest(StrictModel):
    """Compute the preimage of the Euler totient function."""

    target: int = Field(ge=1, le=MAX_N)


class EulerPhiPreimageCountRequest(StrictModel):
    """Count the preimage of the Euler totient function."""

    target: int = Field(ge=1, le=MAX_N)


class EulerPhiPowerSumRequest(StrictModel):
    """Compute sum of k-th powers of preimage of phi."""

    target: int = Field(ge=1, le=MAX_N)
    exponent: int = Field(ge=1, le=20)


# Results


class EulerPhiPreimageResult(StrictModel):
    preimage: tuple[int, ...]
    count: int = Field(ge=0)


class EulerPhiPreimageCountResult(StrictModel):
    count: int = Field(ge=0)


class EulerPhiPowerSumResult(StrictModel):
    power_sum: int = Field(ge=0)
    count: int = Field(ge=0)
