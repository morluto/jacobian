"""Typed contracts for exact Poisson-binomial count distributions."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.probability._distribution import FiniteRationalDistribution
from jacobian.math.probability._models import MAX_RESULT_RATIONAL_DIGITS

MAX_INPUT_RATIONAL_DIGITS = 100
MAX_PROBABILITIES = 255
MAX_INTERMEDIATE_RATIONAL_DIGITS = MAX_RESULT_RATIONAL_DIGITS


class PoissonBinomialAdmissionError(ValueError):
    """Native admission failure for an exact Poisson-binomial computation."""


class PoissonBinomialRequest(StrictModel):
    """A list of canonical rational Bernoulli success probabilities."""

    probabilities: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_PROBABILITIES,
    )


class PoissonBinomialResult(StrictModel):
    """Exact count distribution bound to canonical source probabilities."""

    probabilities: tuple[CanonicalRational, ...]
    count_distribution: FiniteRationalDistribution

    @model_validator(mode="after")
    def bind_count_support(self) -> Self:
        expected_support = tuple(
            Fraction(index) for index in range(len(self.probabilities) + 1)
        )
        actual_support = tuple(
            atom.value.as_fraction() for atom in self.count_distribution.atoms
        )
        if actual_support != expected_support:
            raise ValueError(
                "Poisson-binomial count distribution support must be exactly "
                "0 through n"
            )
        if any(
            probability.as_fraction() < 0 or probability.as_fraction() > 1
            for probability in self.probabilities
        ):
            raise ValueError("probabilities must lie in the closed unit interval")
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PoissonBinomialRequest,
        count_distribution: FiniteRationalDistribution,
    ) -> Self:
        """Construct the result from the trusted admitted recurrence kernel."""

        return cls.model_construct(
            probabilities=request.probabilities,
            count_distribution=count_distribution,
        )


def _admit_probabilities(
    probabilities: tuple[Fraction, ...],
) -> None:
    """Validate the exact native computation envelope."""

    if not 1 <= len(probabilities) <= MAX_PROBABILITIES:
        raise PoissonBinomialAdmissionError(
            "Poisson-binomial probability count must be between 1 and "
            f"{MAX_PROBABILITIES}"
        )
    if any(type(value) is not Fraction for value in probabilities):
        raise TypeError("Poisson-binomial probabilities must use Fractions")
    for value in probabilities:
        if value < 0 or value > 1:
            raise PoissonBinomialAdmissionError(
                "probabilities must lie in the closed unit interval"
            )
        if (
            len(format_canonical_integer(abs(value.numerator)))
            > MAX_INPUT_RATIONAL_DIGITS
            or len(format_canonical_integer(value.denominator))
            > MAX_INPUT_RATIONAL_DIGITS
        ):
            raise PoissonBinomialAdmissionError(
                "Poisson-binomial input probabilities exceed the "
                f"{MAX_INPUT_RATIONAL_DIGITS}-digit bound"
            )
    denominator_digits = sum(
        len(format_canonical_integer(value.denominator)) for value in probabilities
    )
    if denominator_digits > MAX_INTERMEDIATE_RATIONAL_DIGITS:
        raise PoissonBinomialAdmissionError(
            "probability denominators exceed the exact result digit budget of "
            f"{MAX_INTERMEDIATE_RATIONAL_DIGITS} digits"
        )


__all__ = [
    "MAX_INPUT_RATIONAL_DIGITS",
    "MAX_INTERMEDIATE_RATIONAL_DIGITS",
    "MAX_PROBABILITIES",
    "PoissonBinomialRequest",
    "PoissonBinomialResult",
]
