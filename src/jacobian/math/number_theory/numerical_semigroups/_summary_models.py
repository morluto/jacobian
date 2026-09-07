"""Contracts owned by numerical-semigroup summary and membership kernels."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory.numerical_semigroups._models import (
    _GENERAL_ELEMENT_ENVELOPE,
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
)
from jacobian.math.number_theory.numerical_semigroups.values import NumericalSemigroup


class NumericalSemigroupSummaryRequest(StrictModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[ExactInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; the summary uses its increasing minimal generator axis."
        ),
    )


class NumericalSemigroupSummaryResult(StrictModel):
    """Summary of a numerical semigroup."""

    semigroup: NumericalSemigroup

    @property
    def minimal_generators(self) -> tuple[ExactInteger, ...]:
        return self.semigroup.minimal_generators

    multiplicity: ExactInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: ExactInteger
    conductor: ExactInteger
    genus: int = Field(ge=0)
    gaps: tuple[ExactInteger, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        minimal_generators: tuple[ExactInteger, ...],
        multiplicity: ExactInteger,
        embedding_dimension: int,
        frobenius_number: ExactInteger,
        conductor: ExactInteger,
        genus: int,
        gaps: tuple[ExactInteger, ...],
    ) -> NumericalSemigroupSummaryResult:
        """Construct a summary after the native kernel establishes its invariants."""

        return cls.model_construct(
            semigroup=NumericalSemigroup(minimal_generators=minimal_generators),
            multiplicity=multiplicity,
            embedding_dimension=embedding_dimension,
            frobenius_number=frobenius_number,
            conductor=conductor,
            genus=genus,
            gaps=gaps,
        )


class SemigroupMembershipRequest(StrictModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[ExactInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant."
        ),
    )
    value: ExactInteger = Field(
        description="Integer to test for membership. " + _GENERAL_ELEMENT_ENVELOPE
    )


class SemigroupMembershipResult(StrictModel):
    """Whether the value is in the semigroup."""

    value: ExactInteger
    in_semigroup: bool


__all__ = [
    "NumericalSemigroupSummaryRequest",
    "NumericalSemigroupSummaryResult",
    "SemigroupMembershipRequest",
    "SemigroupMembershipResult",
]
