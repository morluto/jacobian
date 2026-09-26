"""Typed contracts for weighted monotone subsequence endpoint profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import decimal_digit_width
from jacobian.math.logic.languages.words.values import FiniteWord

MAX_ENDPOINT_PROFILE_WORK = 250_000
MAX_WEIGHTED_MONOTONE_COMPONENT_DIGITS = 256
MAX_WEIGHTED_MONOTONE_SOURCE_DIGITS = 4_096
MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK = 20_000_000
MAX_WEIGHTED_MONOTONE_RESULT_COMPONENT_DIGITS = 4_096


def _raw_component_digit_width(value: object) -> int | None:
    """Inspect a canonical rational component before Pydantic integer parsing."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return decimal_digit_width(value)
    if isinstance(value, str):
        return len(value.lstrip("-"))
    return None


def _preflight_weight_payload(value: object) -> None:
    """Bound exact rational input spelling before decoding large integers."""

    if not isinstance(value, Mapping):
        return
    weights = value.get("weights")
    if not isinstance(weights, (list, tuple)):
        return
    if len(weights) > 500:
        raise PydanticCustomError(
            "weighted_word.length_exceeded",
            "at most 500 rational weights are admitted",
        )
    total_digits = 0
    for weight in weights:
        if isinstance(weight, CanonicalRational):
            widths = (
                decimal_digit_width(weight.num),
                decimal_digit_width(weight.den),
            )
        elif isinstance(weight, Mapping):
            numerator = _raw_component_digit_width(weight.get("num"))
            denominator = _raw_component_digit_width(weight.get("den"))
            if numerator is None or denominator is None:
                continue
            widths = numerator, denominator
        else:
            continue
        if max(widths) > MAX_WEIGHTED_MONOTONE_COMPONENT_DIGITS:
            raise PydanticCustomError(
                "weighted_word.component_digits",
                "each rational numerator and denominator is limited to 256 decimal digits",
            )
        total_digits += sum(widths)


class WeightedOrderedWord(StrictModel):
    """A finite word with a nonnegative rational weight per position."""

    word: FiniteWord
    weights: tuple[CanonicalRational, ...] = Field(
        max_length=500,
        description=(
            "One nonnegative exact rational per word position. The complete "
            "quadratic profile work and result envelope is checked before execution."
        ),
    )

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if len(self.weights) != len(self.word.letters):
            raise PydanticCustomError(
                "weighted_word.length_mismatch",
                "weights length must match word length",
            )
        for w in self.weights:
            if w.as_fraction() < 0:
                raise PydanticCustomError(
                    "weighted_word.negative_weight",
                    "all weights must be nonnegative",
                )
        return self


class EndpointProfileRequest(StrictModel):
    """Request for the weighted monotone subsequence endpoint profiles."""

    source: WeightedOrderedWord


class EndpointProfileEntry(StrictModel):
    """One position's endpoint values."""

    position: int
    letter: str
    weight: CanonicalRational
    increasing_value: CanonicalRational
    decreasing_value: CanonicalRational


class EndpointProfileResult(StrictModel):
    """The complete endpoint profiles S_i and T_i."""

    source: WeightedOrderedWord
    entries: tuple[EndpointProfileEntry, ...]


WeightedMonotonicity = Literal["NONDECREASING", "NONINCREASING"]


class WeightedMaximumRequest(StrictModel):
    """Request for one exact weighted monotone-subsequence optimum."""

    source: WeightedOrderedWord

    @model_validator(mode="before")
    @classmethod
    def preflight_rational_source(cls, value: object) -> object:
        if isinstance(value, Mapping):
            _preflight_weight_payload(value.get("source"))
        return canonicalize_json_containers(value)


class WeightedMaximumResult(StrictModel):
    """Exact optimum and one source-index witness for a named weak order."""

    source: WeightedOrderedWord
    monotonicity: WeightedMonotonicity
    weight: CanonicalRational
    indices: tuple[int, ...] = Field(max_length=500)
    values: tuple[str, ...] = Field(max_length=500)

    @model_validator(mode="after")
    def validate_witness_structure(self) -> Self:
        if len(self.indices) != len(self.values):
            raise PydanticCustomError(
                "weighted_maximum.witness_shape",
                "indices and values must have equal length",
            )
        if any(
            i < 0 or i >= len(self.source.word.letters) for i in self.indices
        ) or any(a >= b for a, b in zip(self.indices, self.indices[1:], strict=True)):
            raise PydanticCustomError(
                "weighted_maximum.witness_indices",
                "witness indices must be strictly increasing source positions",
            )
        if any(
            self.source.word.letters[i] != value
            for i, value in zip(self.indices, self.values, strict=True)
        ):
            raise PydanticCustomError(
                "weighted_maximum.witness_values",
                "witness values must match the indexed source letters",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        source: WeightedOrderedWord,
        monotonicity: WeightedMonotonicity,
        weight: CanonicalRational,
        indices: tuple[int, ...],
        values: tuple[str, ...],
    ) -> Self:
        """Build a result from the admitted DP without replaying its witness."""
        return cls.model_construct(
            source=source,
            monotonicity=monotonicity,
            weight=weight,
            indices=indices,
            values=values,
        )


__all__ = [
    "MAX_ENDPOINT_PROFILE_WORK",
    "MAX_WEIGHTED_MONOTONE_ARITHMETIC_WORK",
    "MAX_WEIGHTED_MONOTONE_COMPONENT_DIGITS",
    "MAX_WEIGHTED_MONOTONE_RESULT_COMPONENT_DIGITS",
    "MAX_WEIGHTED_MONOTONE_SOURCE_DIGITS",
    "EndpointProfileEntry",
    "EndpointProfileRequest",
    "EndpointProfileResult",
    "WeightedMaximumRequest",
    "WeightedMaximumResult",
    "WeightedMonotonicity",
    "WeightedOrderedWord",
]
