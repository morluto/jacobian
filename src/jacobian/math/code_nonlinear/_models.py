"""Typed wire contracts for nonlinear binary code operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CODEWORDS = 1024
MAX_LENGTH = 16


class BinaryCodeRequest(StrictModel):
    """A binary code as a list of distinct codewords."""

    codewords: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_CODEWORDS
    )

    @model_validator(mode="after")
    def require_valid_codewords(self) -> Self:
        if not self.codewords:
            raise ValueError("codewords must not be empty")
        width = len(self.codewords[0])
        if width == 0 or width > MAX_LENGTH:
            raise ValueError("codeword length must be between 1 and 16")
        if any(len(w) != width for w in self.codewords):
            raise ValueError("all codewords must have equal length")
        if any(b not in (0, 1) for w in self.codewords for b in w):
            raise ValueError("codewords must be binary (0 or 1)")
        if len(set(self.codewords)) != len(self.codewords):
            raise ValueError("codewords must be distinct")
        return self


class ConstantWeightRequest(StrictModel):
    """Generate all constant-weight binary words of given length and weight."""

    length: int = Field(ge=1, le=MAX_LENGTH)
    weight: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_weight(self) -> Self:
        if self.weight > self.length:
            raise ValueError("weight cannot exceed length")
        return self


class DistanceProfileResult(StrictModel):
    minimum_distance: int = Field(ge=0)
    weight_profile: tuple[int, ...]
    method: str = "EXACT_ENUMERATION"


class ConstantWeightResult(StrictModel):
    codewords: tuple[tuple[int, ...], ...]
    count: int = Field(ge=0)
    method: str = "EXACT_ENUMERATION"
