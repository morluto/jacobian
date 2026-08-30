"""Typed contracts for the weighted-antichain operation."""

import math
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.combinatorics.posets.core._models import FinitePoset

MAX_WEIGHTED_ANTICHAIN_ELEMENTS = 16


def _weighted_antichain_admission_error(
    poset: FinitePoset, weights: tuple[CanonicalRational, ...]
) -> tuple[str, str] | None:
    if len(poset.elements) > MAX_WEIGHTED_ANTICHAIN_ELEMENTS:
        return (
            "work_exceeded",
            "weighted antichain search supports at most 16 elements",
        )
    if len(weights) != len(poset.elements):
        return (
            "weight_axis",
            "weights must align one-for-one with the poset element axis",
        )
    nonnegative = [weight for weight in weights if weight.as_fraction() >= 0]
    common_denominator = 1
    for weight in nonnegative:
        denominator = parse_canonical_integer(weight.den)
        common_denominator = (
            common_denominator // math.gcd(common_denominator, denominator)
        ) * denominator
        if (
            len(format_canonical_integer(common_denominator))
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            return (
                "derived_rational_bound",
                "weighted antichain sums exceed the canonical rational digit bound",
            )
    if nonnegative:
        denominator_digits = len(format_canonical_integer(common_denominator))
        numerator_digits = max(
            len(weight.num.lstrip("-"))
            + max(1, denominator_digits - len(weight.den) + 1)
            for weight in nonnegative
        ) + len(str(len(nonnegative)))
        if numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS:
            return (
                "derived_rational_bound",
                "weighted antichain sums exceed the canonical rational digit bound",
            )
    return None


class WeightedAntichainRequest(StrictModel):
    """Request the maximum-weight antichain of a finite poset."""

    poset: FinitePoset
    weights: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_WEIGHTED_ANTICHAIN_ELEMENTS
    )

    @model_validator(mode="after")
    def require_aligned_bounded_weights(self) -> Self:
        failure = _weighted_antichain_admission_error(self.poset, self.weights)
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"poset.weighted_antichain_{code}", message)
        return self


class WeightedAntichainResult(StrictModel):
    """The exact maximum-weight antichain."""

    poset_digest: str
    weights: tuple[CanonicalRational, ...]
    maximum_weight: CanonicalRational
    maximum_antichain: tuple[str, ...]
    method: Literal["EXACT_BOUNDED_SUBSET_SEARCH"]


__all__ = [
    "MAX_WEIGHTED_ANTICHAIN_ELEMENTS",
    "WeightedAntichainRequest",
    "WeightedAntichainResult",
    "_weighted_antichain_admission_error",
]
