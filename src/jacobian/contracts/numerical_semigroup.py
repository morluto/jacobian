"""Typed contracts for numerical semigroup operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel


class NumericalSemigroupRequest(ContractModel):
    """Request for numerical semigroup summary."""

    generators: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=32,
        description="Positive integer generators with gcd one.",
    )

    @model_validator(mode="after")
    def validate_generators(self) -> Self:
        if any(g <= 0 for g in self.generators):
            raise ValueError("generators must be positive")
        if len(set(self.generators)) != len(self.generators):
            raise ValueError("generators must be distinct")
        from math import gcd
        from functools import reduce
        g = reduce(gcd, self.generators)
        if g != 1:
            raise ValueError(f"generators must have gcd 1, got {g}")
        return self


class NumericalSemigroupSummary(ContractModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[StrictInt, ...] = Field(min_length=1)
    multiplicity: StrictInt = Field(gt=0)
    embedding_dimension: StrictInt = Field(gt=0)
    frobenius_number: StrictInt = Field(ge=-1)
    conductor: StrictInt = Field(gt=0)
    genus: StrictInt = Field(ge=0)
    gaps: tuple[StrictInt, ...] = Field(default=())
    aperey_set: tuple[tuple[StrictInt, StrictInt], ...] = Field(default=())

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.multiplicity != min(self.minimal_generators):
            raise ValueError("multiplicity must be the least minimal generator")
        if self.embedding_dimension != len(self.minimal_generators):
            raise ValueError("embedding dimension must equal the number of minimal generators")
        if self.conductor != self.frobenius_number + 1:
            raise ValueError("conductor must be frobenius_number + 1")
        if self.genus != len(self.gaps):
            raise ValueError("genus must equal the number of gaps")
        return self


class NumericalSemigroupMembershipRequest(ContractModel):
    """Request to check membership in a numerical semigroup."""

    generators: tuple[StrictInt, ...] = Field(min_length=1, max_length=32)
    element: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_generators(self) -> Self:
        if any(g <= 0 for g in self.generators):
            raise ValueError("generators must be positive")
        if len(set(self.generators)) != len(self.generators):
            raise ValueError("generators must be distinct")
        from math import gcd
        from functools import reduce
        g = reduce(gcd, self.generators)
        if g != 1:
            raise ValueError(f"generators must have gcd 1, got {g}")
        return self


class NumericalSemigroupMembershipResult(ContractModel):
    """Result of checking membership in a numerical semigroup."""

    is_member: bool
    element: StrictInt = Field(ge=0)
    detail: str = Field(min_length=1, max_length=1024)
