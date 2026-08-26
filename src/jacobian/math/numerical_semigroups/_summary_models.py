"""Contracts owned by numerical-semigroup summary and membership kernels."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.numerical_semigroups._models import (
    _GENERAL_ELEMENT_ENVELOPE,
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    _require_bounded_value,
    _require_canonical_minimal_axis,
    _require_minimal_generators,
    _summary_invariants,
    _validation_error,
)


class NumericalSemigroupSummaryRequest(StrictModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; the summary uses its increasing minimal generator axis."
        ),
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_minimal_generators(self.generators)
        return self


class NumericalSemigroupSummaryResult(StrictModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description="Increasing minimal generator axis of the numerical semigroup.",
    )
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]

    @model_validator(mode="after")
    def require_summary_semantics(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        multiplicity, embedding_dimension, frobenius_number, conductor, genus, gaps = (
            _summary_invariants(generators)
        )
        if parse_canonical_integer(self.multiplicity) != multiplicity:
            raise _validation_error(
                "multiplicity does not match the minimal generators"
            )
        if self.embedding_dimension != embedding_dimension:
            raise _validation_error(
                "embedding_dimension does not match the minimal generators"
            )
        if parse_canonical_integer(self.frobenius_number) != frobenius_number:
            raise _validation_error(
                "frobenius_number does not match the minimal generators"
            )
        if parse_canonical_integer(self.conductor) != conductor:
            raise _validation_error("conductor does not match the minimal generators")
        if self.genus != genus:
            raise _validation_error("genus does not match the minimal generators")
        if tuple(map(parse_canonical_integer, self.gaps)) != gaps:
            raise _validation_error("gaps do not match the minimal generators")
        return self


class SemigroupMembershipRequest(StrictModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant."
        ),
    )
    value: CanonicalInteger = Field(
        description="Integer to test for membership. " + _GENERAL_ELEMENT_ENVELOPE
    )

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_bounded_value(generators, self.value)
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
