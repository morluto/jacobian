"""Typed wire contracts for coding theory operations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from jacobian.contracts.base import ContractModel


class LinearCodeRequest(ContractModel):
    """A linear code given by its generator matrix over a finite field."""

    field_order: int = Field(ge=2, le=256)
    generator_matrix: tuple[tuple[tuple[int, int, int], ...], ...] = Field(
        min_length=1, max_length=32
    )


class MinimumDistanceResult(ContractModel):
    minimum_distance: int = Field(ge=0, le=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"


class WeightDistributionResult(ContractModel):
    weights: tuple[tuple[int, int], ...] = Field(max_length=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"
