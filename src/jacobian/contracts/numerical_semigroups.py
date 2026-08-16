"""Typed wire contracts for numerical semigroup operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger

MAX_GENERATORS = 20
MAX_ELEMENT = 10_000_000
MAX_FACTOR_SEARCH = 100


class NumericalSemigroupRequest(ContractModel):
    """A numerical semigroup defined by a finite set of positive generators."""

    generators: tuple[CanonicalInteger, ...] = Field(min_length=1, max_length=MAX_GENERATORS)

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        for g in self.generators:
            if int(g) <= 0:
                raise ValueError("generators must be positive integers")
        return self


class NumericalSemigroupSummaryRequest(ContractModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(min_length=1, max_length=MAX_GENERATORS)


class NumericalSemigroupSummaryResult(ContractModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]


class SemigroupMembershipRequest(ContractModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(min_length=1, max_length=MAX_GENERATORS)
    value: CanonicalInteger


class SemigroupMembershipResult(ContractModel):
    """Whether the value is in the semigroup."""

    value: CanonicalInteger
    in_semigroup: bool


__all__ = [
    "NumericalSemigroupSummaryRequest",
    "NumericalSemigroupSummaryResult",
    "SemigroupMembershipRequest",
    "SemigroupMembershipResult",
]
