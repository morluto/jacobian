"""Typed wire contracts for nonlinear binary code operations."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,return-value"

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CODEWORDS = 4096
MAX_LENGTH = 64


class BinaryCodeRequest(StrictModel):
    """A binary code as a list of distinct codewords."""

    codewords: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_CODEWORDS
    )

    @model_validator(mode="after")
    def require_valid_codewords(self) -> Self:
        if not self.codewords:
            raise ValueError("codewords must not be empty")
        width = len(self.codewords[0])
        if width == 0 or width > MAX_LENGTH:
            raise ValueError(f"codeword length must be between 1 and {MAX_LENGTH}")
        if any(len(w) != width for w in self.codewords):
            raise ValueError("all codewords must have equal length")
        if any(b not in (0, 1) for w in self.codewords for b in w):
            raise ValueError("codewords must be binary (0 or 1)")
        if len(set(self.codewords)) != len(self.codewords):
            raise ValueError("codewords must be distinct")
        return self


class ConstantWeightRequest(StrictModel):
    """Generate all constant-weight binary words of given length and weight."""

    length: int = Field(ge=1, le=MAX_LENGTH)
    weight: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_weight(self) -> Self:
        if self.weight > self.length:
            raise ValueError("weight cannot exceed length")
        return self


class DistanceProfileResult(StrictModel):
    minimum_distance: int = Field(ge=0)
    weight_profile: tuple[int, ...]
    method: str = "EXACT_ENUMERATION"


class ConstantWeightResult(StrictModel):
    codewords: tuple[tuple[int, ...], ...]
    count: int = Field(ge=0)
    method: str = "EXACT_ENUMERATION"


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
    """Compute the complete profile of an explicit binary code."""

    codewords: tuple[tuple[int, ...], ...] = Field(
        min_length=2, max_length=MAX_CODEWORDS
    )

    @model_validator(mode="after")
    def require_valid_codewords(self) -> Self:
        width = len(self.codewords[0])
        if width == 0 or width > MAX_LENGTH:
            raise ValueError(f"codeword length must be between 1 and {MAX_LENGTH}")
        if any(len(w) != width for w in self.codewords):
            raise ValueError("all codewords must have equal length")
        if any(b not in (0, 1) for w in self.codewords for b in w):
            raise ValueError("codewords must be binary (0 or 1)")
        if len(set(self.codewords)) != len(self.codewords):
            raise ValueError("codewords must be distinct")
        return self


class ExplicitProfileResult(StrictModel):
    """Complete profile of an explicit binary code."""

    codewords: tuple[tuple[int, ...], ...]
    length: int = Field(ge=1)
    cardinality: int = Field(ge=1)
    weight_distribution: tuple[int, ...]
    minimum_distance: int = Field(ge=0)
    maximum_distance: int = Field(ge=0)
    distance_histogram: tuple[int, ...]
    min_distance_pair: tuple[int, int] | None = None
    max_distance_pair: tuple[int, int] | None = None

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _explicit_profile

        profile = _explicit_profile(self.codewords)
        if self.weight_distribution != profile["weight_distribution"]:
            raise ValueError("weight_distribution must be exact")
        if self.minimum_distance != profile["minimum_distance"]:
            raise ValueError("minimum_distance must be exact")
        if self.maximum_distance != profile["maximum_distance"]:
            raise ValueError("maximum_distance must be exact")
        if self.distance_histogram != profile["distance_histogram"]:
            raise ValueError("distance_histogram must be exact")
        return self


class ConstantWeightProfileRequest(StrictModel):
    """Profile of a constant-weight binary code."""

    codewords: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_CODEWORDS
    )

    @model_validator(mode="after")
    def require_valid_constant_weight(self) -> Self:
        if not self.codewords:
            raise ValueError("codewords must not be empty")
        width = len(self.codewords[0])
        if width == 0 or width > MAX_LENGTH:
            raise ValueError(f"codeword length must be between 1 and {MAX_LENGTH}")
        if any(len(w) != width for w in self.codewords):
            raise ValueError("all codewords must have equal length")
        if any(b not in (0, 1) for w in self.codewords for b in w):
            raise ValueError("codewords must be binary (0 or 1)")
        if len(set(self.codewords)) != len(self.codewords):
            raise ValueError("codewords must be distinct")
        weight = sum(self.codewords[0])
        if any(sum(w) != weight for w in self.codewords):
            raise ValueError("all codewords must have the same weight")
        return self


class ConstantWeightProfileResult(StrictModel):
    """Profile of a constant-weight binary code."""

    codewords: tuple[tuple[int, ...], ...]
    length: int = Field(ge=1)
    weight: int = Field(ge=0)
    cardinality: int = Field(ge=1)
    minimum_distance: int = Field(ge=0)
    distance_histogram: tuple[int, ...]

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _constant_weight_profile

        profile = _constant_weight_profile(self.codewords)
        if self.minimum_distance != profile["minimum_distance"]:
            raise ValueError("minimum_distance must be exact")
        if self.distance_histogram != profile["distance_histogram"]:
            raise ValueError("distance_histogram must be exact")
        return self


class ToSetSystemRequest(StrictModel):
    """Map codewords to support subsets on coordinate labels."""

    codewords: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_CODEWORDS
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        width = len(self.codewords[0])
        if width == 0 or width > MAX_LENGTH:
            raise ValueError(f"codeword length must be between 1 and {MAX_LENGTH}")
        if any(len(w) != width for w in self.codewords):
            raise ValueError("all codewords must have equal length")
        if any(b not in (0, 1) for w in self.codewords for b in w):
            raise ValueError("codewords must be binary (0 or 1)")
        return self


class ToSetSystemResult(StrictModel):
    """Support subsets for each codeword."""

    length: int = Field(ge=1)
    cardinality: int = Field(ge=1)
    supports: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def bind_supports(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _to_set_system

        supports = _to_set_system(self.supports, self.length, self.cardinality)
        if self.supports != supports:
            raise ValueError("supports must be exact")
        return self
