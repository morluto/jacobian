"""Typed wire contracts for nonlinear binary code operations."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,return-value"

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.code_nonlinear.values import ExplicitBinaryCode

MAX_CODEWORDS = 4096
MAX_LENGTH = 64

# ``code.nonlinear.constant_weight.compute`` materializes every C(length,
# weight) word, so its admitted domain is bounded by the exact binomial
# output size rather than the shared explicit-code length limit.  The
# worst case at MAX_LENGTH is C(64,32) ~ 1.8e18 words, which cannot be
# enumerated; requests whose complete output would exceed MAX_CODEWORDS
# are rejected before any backend work.
MAX_CONSTANT_WEIGHT_WORDS = MAX_CODEWORDS


class BinaryCodeRequest(StrictModel):
    """A canonical explicit binary code, including empty and degenerate codes."""

    code: ExplicitBinaryCode


class ConstantWeightRequest(StrictModel):
    """Generate all constant-weight binary words of given length and weight."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Generate every constant-weight binary word. Coupled admission "
                "rule: the complete output size C(length, weight) must not "
                f"exceed {MAX_CONSTANT_WEIGHT_WORDS} materialized words; requests "
                "whose binomial output is larger are rejected before enumeration. "
                "For example, length=12 weight=6 (924 words) is admitted while "
                "length=64 weight=32 (~1.8e18 words) is not."
            )
        }
    )

    length: int = Field(
        ge=1,
        le=MAX_LENGTH,
        description=(
            f"Word length in [1, {MAX_LENGTH}]. Subject to the coupled rule "
            f"C(length, weight) <= {MAX_CONSTANT_WEIGHT_WORDS}."
        ),
    )
    weight: int = Field(
        ge=0,
        description=(
            f"Hamming weight in [0, length]. Subject to the coupled rule "
            f"C(length, weight) <= {MAX_CONSTANT_WEIGHT_WORDS}."
        ),
    )

    @model_validator(mode="after")
    def require_valid_weight(self) -> Self:
        if self.weight > self.length:
            raise ValueError("weight cannot exceed length")
        expected_words = comb(self.length, self.weight)
        if expected_words > MAX_CONSTANT_WEIGHT_WORDS:
            raise ValueError(
                "constant-weight enumeration would materialize "
                f"{expected_words:,} words (C({self.length},{self.weight})), "
                f"exceeding the {MAX_CONSTANT_WEIGHT_WORDS}-word result bound; "
                "enumerate a smaller length/weight or compose via subsets"
            )
        return self


class DistanceProfileResult(StrictModel):
    code: ExplicitBinaryCode
    minimum_distance: int = Field(ge=0)
    weight_profile: tuple[int, ...]
    method: str = "EXACT_ENUMERATION"

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        codewords = self.code.codewords
        expected_weights = tuple(sum(word) for word in codewords)
        if self.weight_profile != expected_weights:
            raise ValueError("weight_profile must be exact")
        expected_distance = 0
        if len(codewords) > 1:
            expected_distance = min(
                sum(a != b for a, b in zip(left, right, strict=True))
                for index, left in enumerate(codewords)
                for right in codewords[index + 1 :]
            )
        if self.minimum_distance != expected_distance:
            raise ValueError("minimum_distance must be exact")
        return self


class ConstantWeightResult(StrictModel):
    code: ExplicitBinaryCode
    count: int = Field(ge=0)
    method: str = "EXACT_ENUMERATION"

    @model_validator(mode="after")
    def bind_count(self) -> Self:
        if self.count != len(self.code.codewords):
            raise ValueError("count must equal the source code cardinality")
        return self


class WordDistanceRequest(StrictModel):
    """Compute Hamming distance between two equal-length binary words."""

    word1: tuple[int, ...] = Field(min_length=1, max_length=MAX_LENGTH)
    word2: tuple[int, ...] = Field(min_length=1, max_length=MAX_LENGTH)

    @model_validator(mode="after")
    def require_valid_words(self) -> Self:
        if len(self.word1) != len(self.word2):
            raise ValueError("words must have equal length")
        if any(b not in (0, 1) for b in self.word1 + self.word2):
            raise ValueError("words must be binary (0 or 1)")
        return self


class WordDistanceResult(StrictModel):
    """Result of computing Hamming distance between two binary words."""

    word1: tuple[int, ...]
    word2: tuple[int, ...]
    distance: int = Field(ge=0)
    differing_coordinates: tuple[int, ...]
    weight1: int = Field(ge=0)
    weight2: int = Field(ge=0)
    support_intersection: int = Field(ge=0)

    @model_validator(mode="after")
    def bind_distance(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _word_distance

        dist, diff_coords, w1, w2, inter = _word_distance(self.word1, self.word2)
        if self.distance != dist:
            raise ValueError("distance must be the exact Hamming distance")
        if self.differing_coordinates != diff_coords:
            raise ValueError("differing_coordinates must be exact")
        if self.weight1 != w1:
            raise ValueError("weight1 must be the Hamming weight of word1")
        if self.weight2 != w2:
            raise ValueError("weight2 must be the Hamming weight of word2")
        if self.support_intersection != inter:
            raise ValueError("support_intersection must be exact")
        return self


class ExplicitProfileRequest(StrictModel):
    """Compute a complete profile, including empty and singleton codes."""

    code: ExplicitBinaryCode


class ExplicitProfileResult(StrictModel):
    """Complete profile of an explicit binary code."""

    code: ExplicitBinaryCode
    length: int = Field(ge=0)
    cardinality: int = Field(ge=0)
    weight_distribution: tuple[int, ...]
    minimum_distance: int = Field(ge=0)
    maximum_distance: int = Field(ge=0)
    distance_histogram: tuple[int, ...]
    min_distance_pair: tuple[int, int] | None = None
    max_distance_pair: tuple[int, int] | None = None

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _explicit_profile

        profile = _explicit_profile(self.code)
        if self.length != self.code.length or self.cardinality != len(
            self.code.codewords
        ):
            raise ValueError("length and cardinality must restate the source code")
        if self.weight_distribution != profile["weight_distribution"]:
            raise ValueError("weight_distribution must be exact")
        if self.minimum_distance != profile["minimum_distance"]:
            raise ValueError("minimum_distance must be exact")
        if self.maximum_distance != profile["maximum_distance"]:
            raise ValueError("maximum_distance must be exact")
        if self.distance_histogram != profile["distance_histogram"]:
            raise ValueError("distance_histogram must be exact")
        if self.min_distance_pair != profile["min_distance_pair"]:
            raise ValueError("min_distance_pair must be exact")
        if self.max_distance_pair != profile["max_distance_pair"]:
            raise ValueError("max_distance_pair must be exact")
        return self


class ConstantWeightProfileRequest(StrictModel):
    """Profile of a constant-weight binary code."""

    code: ExplicitBinaryCode

    @model_validator(mode="after")
    def require_valid_constant_weight(self) -> Self:
        codewords = self.code.codewords
        if codewords:
            weight = sum(codewords[0])
            if any(sum(w) != weight for w in codewords):
                raise ValueError("all codewords must have the same weight")
        return self


class ConstantWeightProfileResult(StrictModel):
    """Profile of a constant-weight binary code."""

    code: ExplicitBinaryCode
    length: int = Field(ge=0)
    weight: int = Field(ge=0)
    cardinality: int = Field(ge=0)
    minimum_distance: int = Field(ge=0)
    distance_histogram: tuple[int, ...]

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _constant_weight_profile

        profile = _constant_weight_profile(self.code)
        if self.length != self.code.length or self.cardinality != len(
            self.code.codewords
        ):
            raise ValueError("length and cardinality must restate the source code")
        expected_weight = sum(self.code.codewords[0]) if self.code.codewords else 0
        if self.weight != expected_weight:
            raise ValueError("weight must restate the source code")
        if any(sum(word) != expected_weight for word in self.code.codewords):
            raise ValueError("code must be constant-weight")
        if self.minimum_distance != profile["minimum_distance"]:
            raise ValueError("minimum_distance must be exact")
        if self.distance_histogram != profile["distance_histogram"]:
            raise ValueError("distance_histogram must be exact")
        return self


class ToSetSystemRequest(StrictModel):
    """Map codewords to support subsets on coordinate labels."""

    code: ExplicitBinaryCode


class ToSetSystemResult(StrictModel):
    """Support subsets for each codeword."""

    code: ExplicitBinaryCode
    length: int = Field(ge=0)
    cardinality: int = Field(ge=0)
    supports: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def bind_supports(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _to_set_system

        supports = _to_set_system(self.code)
        if self.supports != supports:
            raise ValueError("supports must be exact")
        if self.length != self.code.length or self.cardinality != len(
            self.code.codewords
        ):
            raise ValueError("length and cardinality must restate the source code")
        return self
