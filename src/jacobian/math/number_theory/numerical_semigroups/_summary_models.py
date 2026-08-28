"""Contracts owned by numerical-semigroup summary and membership kernels."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.number_theory.numerical_semigroups._models import (
    _GENERAL_ELEMENT_ENVELOPE,
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        minimal_generators: tuple[CanonicalInteger, ...],
        multiplicity: CanonicalInteger,
        embedding_dimension: int,
        frobenius_number: str,
        conductor: str,
        genus: int,
        gaps: tuple[CanonicalInteger, ...],
    ) -> NumericalSemigroupSummaryResult:
        """Construct a summary after the native kernel establishes its invariants."""

        return cls.model_construct(
            minimal_generators=minimal_generators,
            multiplicity=multiplicity,
            embedding_dimension=embedding_dimension,
            frobenius_number=frobenius_number,
            conductor=conductor,
            genus=genus,
            gaps=gaps,
        )


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
