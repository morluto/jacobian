"""Typed wire contracts for coding theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class LinearCodeRequest(ContractModel):
    """A linear code given by its generator matrix over one bounded prime field."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def require_bounded_prime_field_matrix(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime for this prime-field operation")
        width = len(self.generator_matrix[0])
        if width == 0 or width > 256:
            raise ValueError("generator rows must have between one and 256 entries")
        if any(len(row) != width for row in self.generator_matrix):
            raise ValueError("generator matrix rows must have equal length")
        if any(
            not 0 <= entry < self.field_order
            for row in self.generator_matrix
            for entry in row
        ):
            raise ValueError("generator entries must be canonical field residues")
        if self.field_order ** len(self.generator_matrix) > 65_536:
            raise ValueError("generator matrix exceeds the exact enumeration bound")
        return self


class MinimumDistanceResult(ContractModel):
    minimum_distance: int = Field(ge=0, le=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"


class WeightDistributionResult(ContractModel):
    weights: tuple[tuple[int, int], ...] = Field(max_length=10000)
    method: Literal["EXACT_ENUMERATION"] = "EXACT_ENUMERATION"
