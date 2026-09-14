"""Contracts for exact finite rational-distribution operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from math import gcd
from typing import Literal, Self

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


def _bounded_pair_sum(
    left: Fraction,
    right: Fraction,
    *,
    label: str,
) -> Fraction:
    """Add two nonnegative rationals or refuse the intermediate they form."""

    common = gcd(left.denominator, right.denominator)
    left_denominator = left.denominator // common
    right_denominator = right.denominator // common
    scaled_numerator = (
        abs(left.numerator) * right_denominator
        + abs(right.numerator) * left_denominator
    )
    common_denominator = left_denominator * right.denominator
    if (
        common_denominator >= 10**MAX_FINITE_DISTRIBUTION_SUM_DIGITS
        or scaled_numerator >= 10**MAX_FINITE_DISTRIBUTION_SUM_DIGITS
    ):
        raise _validation_error(
            f"{label} normalization exceeds the "
            f"{MAX_FINITE_DISTRIBUTION_SUM_DIGITS}-digit intermediate bound"
        )
    return left + right


def _bounded_fraction_sum(
    values: tuple[Fraction, ...],
    *,
    label: str,
) -> Fraction:
    """Sum nonnegative rationals without materializing an over-height fraction.

    Two orderings are avoided deliberately:

    * **Source order.** Accumulating in support order makes the answer depend on
      how the caller happened to list its atoms. Masses that cancel to a small
      fraction - `1/(11p)` and `(11p-1... )`-shaped complements sharing a
      denominator - are pushed past the intermediate bound by the partial sum
      that precedes them, so a normalized law whose every moment fits the
      envelope is refused for a reason unrelated to its mathematics.
    * **Growing partial sums.** Combining mutually coprime denominators one at a
      time accumulates their product monotonically.

    So equal denominators are combined exactly first, where no denominator
    growth can occur at all, and the reduced group totals are then merged as a
    balanced pairwise tree rather than a running prefix.
    """

    if not values:
        return Fraction()

    grouped: dict[int, int] = {}
    for value in values:
        denominator = value.denominator
        grouped[denominator] = grouped.get(denominator, 0) + value.numerator

    level = [
        Fraction(numerator, denominator)
        for denominator, numerator in sorted(grouped.items())
        if numerator
    ]
    if not level:
        return Fraction()

    completed = 0
    while len(level) > 1:
        merged: list[Fraction] = []
        for index in range(0, len(level) - 1, 2):
            if completed % 256 == 0:
                request_checkpoint(
                    "during finite-distribution probability normalization"
                )
            completed += 1
            merged.append(
                _bounded_pair_sum(level[index], level[index + 1], label=label)
            )
        if len(level) % 2:
            merged.append(level[-1])
        level = merged
    return level[0]


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
