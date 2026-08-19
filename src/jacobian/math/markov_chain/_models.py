"""Typed wire contracts for Markov chain operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class TransitionMatrixRequest(StrictModel):
    """A finite stochastic transition matrix with rational entries."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=32
    )

    @model_validator(mode="after")
    def require_stochastic_square_matrix(self) -> Self:
        dimension = len(self.matrix)
        if any(len(row) != dimension for row in self.matrix):
            raise ValueError("transition matrix must be square")
        for row in self.matrix:
            values = tuple(value.as_fraction() for value in row)
            if any(value < 0 for value in values):
                raise ValueError("transition probabilities must be nonnegative")
            if sum(values) != 1:
                raise ValueError("each transition row must sum to one")
        return self

    @model_validator(mode="after")
    def preflight_stationary_rational_height(self) -> Self:
        """Reject requests whose stationary-distribution rational height
        conservatively exceeds the canonical-rational digit limit.

        For a kxk closed-class submatrix, the stationary distribution solves
        an integer linear system whose determinant bounds the denominator
        height.  By Hadamard's bound, |det(A)| <= (k * max_entry^2)^(k/2)
        where max_entry is the largest absolute value in the cleared integer
        matrix.  We use this to derive a conservative digit bound.
        """
        from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS

        dimension = len(self.matrix)
        # Find the common denominator of all entries
        max_den = 1
        for row in self.matrix:
            for value in row:
                den = value.den.lstrip("-")
                if len(den) > max_den:
                    # Track the max denominator digit count, not the value
                    pass
        # Each entry has numerator/denominator of at most D digits
        max_digit_count = max(
            len(value.num.lstrip("-")) for row in self.matrix for value in row
        ) + max(len(value.den.lstrip("-")) for row in self.matrix for value in row)
        # After clearing denominators, the integer matrix entries have
        # height at most (max_entry)^dimension.  By Hadamard's bound,
        # det(A) has at most dimension * max_digit_count * dimension digits.
        # This is a very conservative bound.
        hadamard_digit_bound = dimension * dimension * max_digit_count + 1
        if hadamard_digit_bound > MAX_CANONICAL_RATIONAL_DIGITS:
            raise ValueError(
                "stationary distribution rational height conservatively exceeds "
                "the canonical-rational digit limit; the transition matrix entries "
                "have denominators too large for exact stationary solving"
            )
        return self


class ExtremeStationaryDistribution(StrictModel):
    """One canonical extreme point supported on a closed class."""

    closed_class: tuple[int, ...] = Field(min_length=1)
    distribution: tuple[CanonicalRational, ...] = Field(min_length=1)


class StationaryDistributionResult(StrictModel):
    """Extreme points of the finite chain's stationary-distribution simplex."""

    extreme_distributions: tuple[ExtremeStationaryDistribution, ...] = Field(
        min_length=1
    )
    unique: bool
    method: Literal["CLOSED_CLASS_EXACT_LINEAR_SYSTEM"] = (
        "CLOSED_CLASS_EXACT_LINEAR_SYSTEM"
    )

    @model_validator(mode="after")
    def bind_stationary_family(self) -> Self:
        dimensions = {len(item.distribution) for item in self.extreme_distributions}
        if len(dimensions) != 1:
            raise ValueError("stationary distributions must share one dimension")
        classes = tuple(item.closed_class for item in self.extreme_distributions)
        if classes != tuple(sorted(classes)) or len(classes) != len(set(classes)):
            raise ValueError("closed classes must be unique and sorted")
        if self.unique != (len(self.extreme_distributions) == 1):
            raise ValueError("unique must match the number of extreme distributions")
        for item in self.extreme_distributions:
            values = tuple(value.as_fraction() for value in item.distribution)
            if any(value < 0 for value in values) or sum(values) != 1:
                raise ValueError(
                    "each stationary distribution must be nonnegative and normalized"
                )
            support = tuple(index for index, value in enumerate(values) if value > 0)
            if support != item.closed_class:
                raise ValueError(
                    "each extreme distribution must be supported on its closed class"
                )
        return self


class ErgodicDecisionResult(StrictModel):
    is_ergodic: bool
    is_irreducible: bool
    is_aperiodic: bool
