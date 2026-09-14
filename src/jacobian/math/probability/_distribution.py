"""Contracts for exact finite rational-distribution operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from math import gcd
from typing import Final, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.math.probability._models import (
    MAX_INPUT_RATIONAL_DIGITS,
    MAX_RESULT_RATIONAL_DIGITS,
    _require_strictly_increasing,
    _validation_error,
)

MAX_FINITE_INPUT_ATOMS = 256
MAX_FINITE_DISTRIBUTION_ATOMS = 32_768
MAX_FINITE_CONVOLUTION_PAIRS = 4096
MAX_FINITE_CONVOLUTION_OUTPUT_ATOMS = 256
MAX_FINITE_CONVOLUTION_POWER = 10**15
MAX_FINITE_DISTRIBUTION_SUM_DIGITS = MAX_RESULT_RATIONAL_DIGITS


def _remove_prime_power(value: int, prime: int) -> int:
    if value % prime:
        return value
    powers = [prime]
    square = prime * prime
    while value % square == 0:
        powers.append(square)
        if square > value // square:
            break
        square *= square
    for power in reversed(powers):
        if value % power == 0:
            value //= power
    return value


_SMALL_KERNEL_PRIMES: Final = tuple(
    candidate
    for candidate in range(2, 1_000)
    if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1))
)


def _large_denominator_kernel(value: int) -> int:
    """Return the part of a denominator left after removing small prime factors.

    Charges that share a large factor (for example ``2p``, ``3p``, and ``6p``)
    collapse to one group so their exact sum can cancel before unrelated groups
    are combined. Removing a general small-prime set (rather than only two and
    three) also pairs masses whose reduction cancelled a shared middle prime,
    such as ``1/(5p)`` and ``(p-1)/(5p)``.
    """

    kernel = abs(value) or 1
    for prime in _SMALL_KERNEL_PRIMES:
        if prime * prime > kernel:
            break
        kernel = _remove_prime_power(kernel, prime)
    return kernel


_SUM_VALUE_LIMIT = 10**MAX_FINITE_DISTRIBUTION_SUM_DIGITS
# Above this many distinct buckets the all-pairs merge scan is replaced by a
# bounded linear accumulation, so a permitted request cannot monopolize a
# worker with an exhaustive GCD search.
_MAX_GREEDY_MERGE_TERMS = 64


def _raise_normalization_bound(label: str) -> None:
    raise _validation_error(
        f"{label} normalization exceeds the "
        f"{MAX_FINITE_DISTRIBUTION_SUM_DIGITS}-digit intermediate bound"
    )


def _add_height_bounded(left: Fraction, right: Fraction, *, label: str) -> Fraction:
    """Add two nonnegative rationals after bounding common-denominator growth."""

    if right == 0:
        return left
    if left == 0:
        return right
    common = gcd(left.denominator, right.denominator)
    left_scale = right.denominator // common
    right_scale = left.denominator // common
    if (
        left_scale > _SUM_VALUE_LIMIT // max(1, abs(left.numerator))
        or right_scale > _SUM_VALUE_LIMIT // max(1, abs(right.numerator))
        or left_scale > _SUM_VALUE_LIMIT // left.denominator
    ):
        _raise_normalization_bound(label)
    numerator = left.numerator * left_scale + right.numerator * right_scale
    denominator = left.denominator * left_scale
    if abs(numerator) >= _SUM_VALUE_LIMIT or denominator >= _SUM_VALUE_LIMIT:
        _raise_normalization_bound(label)
    return Fraction(numerator, denominator)


def _bounded_fraction_sum(
    values: tuple[Fraction, ...],
    *,
    label: str,
) -> Fraction:
    """Sum nonnegative rationals without paying source-order LCD growth.

    Masses are bucketed by the small-prime-free kernel of each denominator so
    complementary pairs such as ``1/(5p)`` and ``(p-1)/(5p)`` reduce before
    unrelated primes are combined. Unique reduced denominators are then
    digit-budgeted, and each addition bounds its cancelled scales before any
    common-denominator multiply, so a unit-sum law cannot force unbounded GCD
    work.
    """

    buckets: dict[int, Fraction] = {}
    for index, value in enumerate(values):
        if index % 128 == 0:
            request_checkpoint("during finite-distribution normalization")
        kernel = _large_denominator_kernel(value.denominator)
        buckets[kernel] = _add_height_bounded(
            buckets.get(kernel, Fraction()),
            value,
            label=label,
        )
    reduced = list(buckets.values())
    return _merge_terms_by_largest_gcd(reduced, label=label)


def _merge_terms_by_largest_gcd(
    terms: list[Fraction],
    *,
    label: str,
) -> Fraction:
    """Combine exact rationals by repeatedly merging the largest shared factor.

    Complementary masses need not reduce through a fixed small-prime set: any
    two denominators sharing a prime can cancel. Merging the pair with the
    largest gcd first keeps intermediate denominators minimal, so a law whose
    pairs reduce through a large shared factor is admitted without a hard-coded
    cutoff, while genuinely unrelated denominators still hit the digit bound.
    """

    pool = [term for term in terms if term != 0]
    while len(pool) > 1 and len(pool) <= _MAX_GREEDY_MERGE_TERMS:
        request_checkpoint("during finite-distribution cancellation")
        best_left = -1
        best_right = -1
        best_shared = 1
        for left in range(len(pool)):
            for right in range(left + 1, len(pool)):
                shared = gcd(pool[left].denominator, pool[right].denominator)
                if shared > best_shared:
                    best_shared = shared
                    best_left, best_right = left, right
        if best_left < 0:
            break
        merged = _add_height_bounded(pool[best_left], pool[best_right], label=label)
        pool = [
            term
            for index, term in enumerate(pool)
            if index not in (best_left, best_right)
        ]
        if merged:
            pool.append(merged)
    total = Fraction()
    for term in pool:
        total = _add_height_bounded(total, term, label=label)
    return total


class FiniteDistributionAtom(StrictModel):
    value: CanonicalRational
    probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_probability(self) -> Self:
        require_bounded_rational(
            self.value,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite-distribution atom",
        )
        require_bounded_rational(
            self.probability,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite-distribution probability",
        )
        if self.probability.as_fraction() < 0:
            raise _validation_error(
                "finite-distribution probabilities must be nonnegative"
            )
        return self


class FiniteRationalDistribution(StrictModel):
    atoms: tuple[FiniteDistributionAtom, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def require_canonical_probability_distribution(self) -> Self:
        _require_strictly_increasing(
            tuple(atom.value for atom in self.atoms),
            label="finite-distribution support values",
        )
        return self


def require_input_distribution(
    atoms: tuple[FiniteDistributionAtom, ...],
    *,
    require_canonical: bool,
    max_digits: int | None = MAX_INPUT_RATIONAL_DIGITS,
) -> tuple[Fraction, ...]:
    values = tuple(atom.value.as_fraction() for atom in atoms)
    if len(values) != len(set(values)):
        raise _validation_error("finite-distribution support values must be unique")
    if require_canonical and any(left >= right for left, right in pairwise(values)):
        raise _validation_error(
            "finite-distribution support values must be strictly increasing"
        )
    if max_digits is not None:
        for atom in atoms:
            require_bounded_rational(
                atom.value,
                max_digits=max_digits,
                label="finite-distribution input atom",
            )
            require_bounded_rational(
                atom.probability,
                max_digits=max_digits,
                label="finite-distribution input probability",
            )
    if (
        _bounded_fraction_sum(
            tuple(atom.probability.as_fraction() for atom in atoms),
            label="finite-distribution input probability",
        )
        != 1
    ):
        raise _validation_error(
            "finite-distribution probabilities must sum exactly to 1"
        )
    return values


class FiniteRawMomentRequest(StrictModel):
    atoms: tuple[FiniteDistributionAtom, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_INPUT_ATOMS,
    )
    order: StrictInt = Field(ge=0, le=128)


class FiniteRawMomentContribution(StrictModel):
    value: CanonicalRational
    probability: CanonicalRational
    powered_value: CanonicalRational
    contribution: CanonicalRational


class FiniteRawMomentResult(StrictModel):
    order: StrictInt = Field(ge=0, le=128)
    moment: CanonicalRational
    contributions: tuple[FiniteRawMomentContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_INPUT_ATOMS,
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        order: int,
        moment: CanonicalRational,
        contributions: tuple[FiniteRawMomentContribution, ...],
    ) -> Self:
        return cls.model_construct(
            order=order, moment=moment, contributions=contributions
        )


class FiniteEventRequest(StrictModel):
    distribution: FiniteRationalDistribution
    event_values: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_FINITE_INPUT_ATOMS
    )


class FiniteConditionRequest(FiniteEventRequest):
    """A finite event known to have positive exact probability."""


class FiniteEventProbabilityResult(StrictModel):
    event_probability: CanonicalRational
    selected_atoms: tuple[FiniteDistributionAtom, ...] = Field(
        max_length=MAX_FINITE_INPUT_ATOMS
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        event_probability: CanonicalRational,
        selected_atoms: tuple[FiniteDistributionAtom, ...],
    ) -> Self:
        return cls.model_construct(
            event_probability=event_probability,
            selected_atoms=selected_atoms,
        )


class FiniteConditionalContribution(StrictModel):
    value: CanonicalRational
    source_probability: CanonicalRational
    conditioned_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_masses(self) -> Self:
        for label, value in (
            ("conditional value", self.value),
            ("conditional source probability", self.source_probability),
            ("conditioned probability", self.conditioned_probability),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        if (
            self.source_probability.as_fraction() < 0
            or self.conditioned_probability.as_fraction() < 0
        ):
            raise _validation_error(
                "conditional contribution masses must be nonnegative"
            )
        return self


class FiniteConditionResult(StrictModel):
    event_probability: CanonicalRational
    distribution: FiniteRationalDistribution
    contributions: tuple[FiniteConditionalContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_INPUT_ATOMS,
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        event_probability: CanonicalRational,
        distribution: FiniteRationalDistribution,
        contributions: tuple[FiniteConditionalContribution, ...],
    ) -> Self:
        return cls.model_construct(
            event_probability=event_probability,
            distribution=distribution,
            contributions=contributions,
        )


class FinitePushforwardMapEntry(StrictModel):
    source: CanonicalRational
    target: CanonicalRational


class FinitePushforwardRequest(StrictModel):
    distribution: FiniteRationalDistribution
    mapping: tuple[FinitePushforwardMapEntry, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_INPUT_ATOMS,
    )


class FinitePushforwardContribution(StrictModel):
    source: CanonicalRational
    target: CanonicalRational
    probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_mass(self) -> Self:
        for label, value in (
            ("pushforward source", self.source),
            ("pushforward target", self.target),
            ("pushforward probability", self.probability),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        if self.probability.as_fraction() < 0:
            raise _validation_error("pushforward contribution mass must be nonnegative")
        return self


class FinitePushforwardResult(StrictModel):
    distribution: FiniteRationalDistribution
    contributions: tuple[FinitePushforwardContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_INPUT_ATOMS,
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        distribution: FiniteRationalDistribution,
        contributions: tuple[FinitePushforwardContribution, ...],
    ) -> Self:
        return cls.model_construct(
            distribution=distribution,
            contributions=contributions,
        )


class FiniteConvolutionRequest(StrictModel):
    left: FiniteRationalDistribution
    right: FiniteRationalDistribution


class FiniteConvolutionPowerRequest(StrictModel):
    """One positive i.i.d. convolution exponent over an exact source law."""

    distribution: FiniteRationalDistribution
    exponent: StrictInt = Field(ge=1, le=MAX_FINITE_CONVOLUTION_POWER)


class FiniteConvolutionPowerResult(StrictModel):
    """The complete exact law of an i.i.d. sum, bound to its source."""

    source: FiniteRationalDistribution
    exponent: StrictInt = Field(ge=1, le=MAX_FINITE_CONVOLUTION_POWER)
    distribution: FiniteRationalDistribution

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRationalDistribution,
        exponent: int,
        distribution: FiniteRationalDistribution,
    ) -> Self:
        return cls.model_construct(
            source=source,
            exponent=exponent,
            distribution=distribution,
        )


class FiniteConvolutionPeakResult(StrictModel):
    """Every maximizer and the exact largest mass of an i.i.d. sum."""

    source: FiniteRationalDistribution
    exponent: StrictInt = Field(ge=1, le=MAX_FINITE_CONVOLUTION_POWER)
    maximum_probability: CanonicalRational
    maximizing_values: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def require_canonical_peak_shape(self) -> Self:
        require_bounded_rational(
            self.maximum_probability,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="convolution-power maximum probability",
        )
        if self.maximum_probability.as_fraction() <= 0:
            raise _validation_error(
                "convolution-power maximum probability must be positive"
            )
        _require_strictly_increasing(
            self.maximizing_values,
            label="convolution-power maximizing values",
        )
        for value in self.maximizing_values:
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="convolution-power maximizing value",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: FiniteRationalDistribution,
        exponent: int,
        maximum_probability: CanonicalRational,
        maximizing_values: tuple[CanonicalRational, ...],
    ) -> Self:
        return cls.model_construct(
            source=source,
            exponent=exponent,
            maximum_probability=maximum_probability,
            maximizing_values=maximizing_values,
        )


class FiniteConvolutionContribution(StrictModel):
    left_value: CanonicalRational
    right_value: CanonicalRational
    sum_value: CanonicalRational
    probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_nonnegative_mass(self) -> Self:
        for label, value in (
            ("convolution left value", self.left_value),
            ("convolution right value", self.right_value),
            ("convolution sum value", self.sum_value),
            ("convolution probability", self.probability),
        ):
            require_bounded_rational(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label=label,
            )
        if self.probability.as_fraction() < 0:
            raise _validation_error("convolution contribution mass must be nonnegative")
        return self


class FiniteConvolutionResult(StrictModel):
    distribution: FiniteRationalDistribution
    contributions: tuple[FiniteConvolutionContribution, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_CONVOLUTION_PAIRS,
    )
    independence: Literal["PRODUCT_MEASURE"] = "PRODUCT_MEASURE"

    @classmethod
    def _from_kernel(
        cls,
        *,
        distribution: FiniteRationalDistribution,
        contributions: tuple[FiniteConvolutionContribution, ...],
    ) -> Self:
        return cls.model_construct(
            distribution=distribution,
            contributions=contributions,
            independence="PRODUCT_MEASURE",
        )


__all__ = [
    "MAX_FINITE_CONVOLUTION_OUTPUT_ATOMS",
    "MAX_FINITE_CONVOLUTION_PAIRS",
    "MAX_FINITE_CONVOLUTION_POWER",
    "MAX_FINITE_DISTRIBUTION_ATOMS",
    "MAX_FINITE_DISTRIBUTION_SUM_DIGITS",
    "MAX_FINITE_INPUT_ATOMS",
    "FiniteConditionRequest",
    "FiniteConditionResult",
    "FiniteConditionalContribution",
    "FiniteConvolutionContribution",
    "FiniteConvolutionPeakResult",
    "FiniteConvolutionPowerRequest",
    "FiniteConvolutionPowerResult",
    "FiniteConvolutionRequest",
    "FiniteConvolutionResult",
    "FiniteDistributionAtom",
    "FiniteEventProbabilityResult",
    "FiniteEventRequest",
    "FinitePushforwardContribution",
    "FinitePushforwardMapEntry",
    "FinitePushforwardRequest",
    "FinitePushforwardResult",
    "FiniteRationalDistribution",
    "FiniteRawMomentContribution",
    "FiniteRawMomentRequest",
    "FiniteRawMomentResult",
    "require_input_distribution",
]
