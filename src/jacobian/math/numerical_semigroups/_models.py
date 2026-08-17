"""Typed wire contracts for numerical semigroup operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

MAX_GENERATORS = 20
MAX_GENERATOR = 500
MAX_ELEMENT = 10_000
MAX_FACTOR_SEARCH = 100


def _require_positive_bounded_generators(generators: tuple[str, ...]) -> None:
    values: list[int] = []
    for generator in generators:
        value = parse_canonical_integer(generator)
        if value <= 0:
            raise ValueError("generators must be positive integers")
        if value > MAX_GENERATOR:
            raise ValueError(f"generators must be at most {MAX_GENERATOR}")
        values.append(value)
    gcd = values[0]
    for value in values[1:]:
        while value:
            gcd, value = value, gcd % value
    if gcd != 1:
        raise ValueError(f"generators must have gcd 1, got gcd {gcd}")


class NumericalSemigroupRequest(StrictModel):
    """A numerical semigroup defined by a finite set of positive generators."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryRequest(StrictModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryResult(StrictModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]


class SemigroupMembershipRequest(StrictModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"membership value must be at most {MAX_ELEMENT}")
        return self


class SemigroupMembershipResult(StrictModel):
    """Whether the value is in the semigroup."""

    value: CanonicalInteger
    in_semigroup: bool


__all__ = [
    "NumericalSemigroupSummaryRequest",
    "NumericalSemigroupSummaryResult",
    "SemigroupMembershipRequest",
    "SemigroupMembershipResult",
]
