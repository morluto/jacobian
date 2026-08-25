"""Contracts for exact finite rational-distribution operations."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.probability._models import (
    MAX_INPUT_RATIONAL_DIGITS,
    MAX_RESULT_RATIONAL_DIGITS,
    _require_bounded_fraction,
    _require_strictly_increasing,
    _validation_error,
)

MAX_FINITE_DISTRIBUTION_ATOMS = 256
MAX_FINITE_CONVOLUTION_PAIRS = 4096


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
        if (
            sum(
                (atom.probability.as_fraction() for atom in self.atoms),
                start=Fraction(),
            )
            != 1
        ):
            raise _validation_error(
                "finite-distribution probabilities must sum exactly to 1"
            )
        return self


def require_input_distribution(
    atoms: tuple[FiniteDistributionAtom, ...],
    *,
    require_canonical: bool,
) -> tuple[Fraction, ...]:
    values = tuple(atom.value.as_fraction() for atom in atoms)
    if len(values) != len(set(values)):
        raise _validation_error("finite-distribution support values must be unique")
    if require_canonical and any(left >= right for left, right in pairwise(values)):
        raise _validation_error(
            "finite-distribution support values must be strictly increasing"
        )
    for atom in atoms:
        require_bounded_rational(
            atom.value,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="finite-distribution input atom",
        )
        require_bounded_rational(
            atom.probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="finite-distribution input probability",
        )
    if (
        sum(
            (atom.probability.as_fraction() for atom in atoms),
            start=Fraction(),
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
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )
    order: StrictInt = Field(ge=0, le=128)

    @model_validator(mode="after")
    def require_probability_distribution(self) -> Self:
        require_input_distribution(self.atoms, require_canonical=False)
        return self


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
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def bind_exact_contributions(self) -> Self:
        total = Fraction()
        for item in self.contributions:
            expected_power = item.value.as_fraction() ** self.order
            if item.powered_value.as_fraction() != expected_power:
                raise _validation_error("moment powered value does not match its atom")
            expected_contribution = item.probability.as_fraction() * expected_power
            if item.contribution.as_fraction() != expected_contribution:
                raise _validation_error("moment contribution does not match its atom")
            total += expected_contribution
        if self.moment.as_fraction() != total:
            raise _validation_error(
                "moment does not equal the sum of atom contributions"
            )
        return self


class FiniteEventRequest(StrictModel):
    distribution: FiniteRationalDistribution
    event_values: tuple[CanonicalRational, ...] = Field(
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS
    )

    @model_validator(mode="after")
    def require_explicit_support_subset(self) -> Self:
        support = set(
            require_input_distribution(
                self.distribution.atoms,
                require_canonical=True,
            )
        )
        event = _require_strictly_increasing(
            self.event_values,
            label="finite event values",
        )
        for value in self.event_values:
            require_bounded_rational(
                value,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="finite event value",
            )
        if not set(event).issubset(support):
            raise _validation_error(
                "finite event values must belong to the distribution"
            )
        event_mass = sum(
            (
                atom.probability.as_fraction()
                for atom in self.distribution.atoms
                if atom.value.as_fraction() in set(event)
            ),
            start=Fraction(),
        )
        _require_bounded_fraction(
            event_mass,
            max_digits=MAX_RESULT_RATIONAL_DIGITS,
            label="finite event probability",
        )
        return self


class FiniteConditionRequest(FiniteEventRequest):
    """A finite event known to have positive exact probability."""

    @model_validator(mode="after")
    def require_positive_event_mass(self) -> Self:
        selected = {value.as_fraction() for value in self.event_values}
        mass = sum(
            (
                atom.probability.as_fraction()
                for atom in self.distribution.atoms
                if atom.value.as_fraction() in selected
            ),
            start=Fraction(),
        )
        if mass <= 0:
            raise _validation_error(
                "conditioning requires a positive-mass finite event"
            )
        return self


class FiniteEventProbabilityResult(StrictModel):
    event_probability: CanonicalRational
    selected_atoms: tuple[FiniteDistributionAtom, ...] = Field(
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS
    )

    @model_validator(mode="after")
    def bind_selected_atom_contributions(self) -> Self:
        _require_strictly_increasing(
            tuple(atom.value for atom in self.selected_atoms),
            label="selected finite-event atoms",
        )
        total = sum(
            (atom.probability.as_fraction() for atom in self.selected_atoms),
            start=Fraction(),
        )
        if self.event_probability.as_fraction() != total:
            raise _validation_error(
                "event probability does not equal selected atom mass"
            )
        return self


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
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def bind_normalized_contributions(self) -> Self:
        event_probability = self.event_probability.as_fraction()
        if event_probability <= 0:
            raise _validation_error(
                "conditional distribution requires positive event mass"
            )
        values = tuple(item.value for item in self.contributions)
        _require_strictly_increasing(
            values,
            label="conditional contribution values",
        )
        expected_atoms: list[tuple[Fraction, Fraction]] = []
        source_total = Fraction()
        for item in self.contributions:
            source = item.source_probability.as_fraction()
            conditioned = item.conditioned_probability.as_fraction()
            if source < 0 or conditioned != source / event_probability:
                raise _validation_error(
                    "conditioned probability does not match source mass"
                )
            source_total += source
            expected_atoms.append((item.value.as_fraction(), conditioned))
        if source_total != event_probability:
            raise _validation_error("conditional contributions do not equal event mass")
        actual_atoms = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual_atoms != expected_atoms:
            raise _validation_error(
                "conditional distribution does not match contributions"
            )
        return self


class FinitePushforwardMapEntry(StrictModel):
    source: CanonicalRational
    target: CanonicalRational


class FinitePushforwardRequest(StrictModel):
    distribution: FiniteRationalDistribution
    mapping: tuple[FinitePushforwardMapEntry, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def require_total_canonical_lookup(self) -> Self:
        source_values = require_input_distribution(
            self.distribution.atoms,
            require_canonical=True,
        )
        mapping_sources = tuple(item.source.as_fraction() for item in self.mapping)
        if mapping_sources != source_values:
            raise _validation_error(
                "pushforward mapping must cover each source atom in canonical order"
            )
        aggregated: dict[Fraction, Fraction] = {}
        for atom, item in zip(self.distribution.atoms, self.mapping, strict=True):
            require_bounded_rational(
                item.source,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="pushforward source",
            )
            require_bounded_rational(
                item.target,
                max_digits=MAX_INPUT_RATIONAL_DIGITS,
                label="pushforward target",
            )
            target = item.target.as_fraction()
            aggregated[target] = (
                aggregated.get(target, Fraction()) + atom.probability.as_fraction()
            )
        for target, probability in aggregated.items():
            _require_bounded_fraction(
                target,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="pushforward target",
            )
            _require_bounded_fraction(
                probability,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="pushforward probability",
            )
        return self


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
        max_length=MAX_FINITE_DISTRIBUTION_ATOMS,
    )

    @model_validator(mode="after")
    def bind_aggregated_pushforward(self) -> Self:
        _require_strictly_increasing(
            tuple(item.source for item in self.contributions),
            label="pushforward contribution sources",
        )
        aggregated: dict[Fraction, Fraction] = {}
        for item in self.contributions:
            target = item.target.as_fraction()
            probability = item.probability.as_fraction()
            aggregated[target] = aggregated.get(target, Fraction()) + probability
        expected = sorted(aggregated.items())
        actual = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual != expected:
            raise _validation_error(
                "pushforward distribution does not match contributions"
            )
        return self


class FiniteConvolutionRequest(StrictModel):
    left: FiniteRationalDistribution
    right: FiniteRationalDistribution

    @model_validator(mode="after")
    def require_bounded_pair_product(self) -> Self:
        require_input_distribution(self.left.atoms, require_canonical=True)
        require_input_distribution(self.right.atoms, require_canonical=True)
        pair_count = len(self.left.atoms) * len(self.right.atoms)
        if pair_count > MAX_FINITE_CONVOLUTION_PAIRS:
            raise _validation_error(
                "finite convolution exceeds the "
                f"{MAX_FINITE_CONVOLUTION_PAIRS}-pair bound"
            )
        aggregated: dict[Fraction, Fraction] = {}
        for left in self.left.atoms:
            for right in self.right.atoms:
                value = left.value.as_fraction() + right.value.as_fraction()
                probability = (
                    left.probability.as_fraction() * right.probability.as_fraction()
                )
                aggregated[value] = aggregated.get(value, Fraction()) + probability
        if len(aggregated) > MAX_FINITE_DISTRIBUTION_ATOMS:
            raise _validation_error(
                "finite convolution exceeds the "
                f"{MAX_FINITE_DISTRIBUTION_ATOMS}-atom output bound"
            )
        for value, probability in aggregated.items():
            _require_bounded_fraction(
                value,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="convolution atom",
            )
            _require_bounded_fraction(
                probability,
                max_digits=MAX_RESULT_RATIONAL_DIGITS,
                label="convolution probability",
            )
        return self


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

    @model_validator(mode="after")
    def bind_aggregated_pairs(self) -> Self:
        aggregated: dict[Fraction, Fraction] = {}
        previous: tuple[Fraction, Fraction] | None = None
        for item in self.contributions:
            left = item.left_value.as_fraction()
            right = item.right_value.as_fraction()
            pair = (left, right)
            if previous is not None and pair <= previous:
                raise _validation_error(
                    "convolution contributions must use canonical pair order"
                )
            previous = pair
            value = item.sum_value.as_fraction()
            if value != left + right:
                raise _validation_error("convolution sum value does not match its pair")
            probability = item.probability.as_fraction()
            aggregated[value] = aggregated.get(value, Fraction()) + probability
        expected = sorted(aggregated.items())
        actual = [
            (atom.value.as_fraction(), atom.probability.as_fraction())
            for atom in self.distribution.atoms
        ]
        if actual != expected:
            raise _validation_error(
                "convolution distribution does not match pair contributions"
            )
        return self


__all__ = [
    "MAX_FINITE_CONVOLUTION_PAIRS",
    "MAX_FINITE_DISTRIBUTION_ATOMS",
    "FiniteConditionRequest",
    "FiniteConditionResult",
    "FiniteConditionalContribution",
    "FiniteConvolutionContribution",
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
