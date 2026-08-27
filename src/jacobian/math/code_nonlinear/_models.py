"""Typed wire contracts for nonlinear binary code operations."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.code_nonlinear._budget import (
    MAX_PROFILE_PAIRS,
)
from jacobian.math.code_nonlinear.values import (
    MAX_EXPLICIT_CODE_LENGTH,
    BinaryWord,
    ExplicitBinaryCode,
)

StrictNonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]
Coordinate = Annotated[
    int,
    Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH - 1),
]
ExplicitLength = Annotated[
    int,
    Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH),
]


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


class ConstantWeightRequest(StrictModel):
    """Generate all binary words of one bounded length and weight."""

    length: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]
    weight: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]


class ConstantWeightResult(StrictModel):
    """Complete generated constant-weight code.

    Construction checks the declared structural envelope only. The owner
    kernel uses ``_from_kernel``; an explicit owner-local verifier checks a
    claim supplied independently of that kernel.
    """

    length: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]
    weight: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]
    code: ExplicitBinaryCode
    count: StrictPositiveInt

    @model_validator(mode="after")
    def require_structural_envelope(self) -> Self:
        if self.code.length != self.length:
            raise _validation_error(
                "nonlinear_code.length_mismatch",
                "code length must equal the declared length",
            )
        if self.count != len(self.code.codewords):
            raise _validation_error(
                "nonlinear_code.cardinality_mismatch",
                "count must equal the retained code cardinality",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, length: int, weight: int, code: ExplicitBinaryCode
    ) -> Self:
        return cls.model_construct(
            length=length, weight=weight, code=code, count=len(code.codewords)
        )


class WordDistanceRequest(StrictModel):
    """Compute Hamming data for two equal-length materialized binary words."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "Two nonempty equal-length binary words of strict integer bits "
                "0 or 1. The exact result, including every differing coordinate, "
                "must fit the 4194304-byte result bound."
            )
        },
    )

    word1: BinaryWord = Field(min_length=1, max_length=MAX_EXPLICIT_CODE_LENGTH)
    word2: BinaryWord = Field(min_length=1, max_length=MAX_EXPLICIT_CODE_LENGTH)

    @model_validator(mode="after")
    def require_valid_words(self) -> Self:
        if len(self.word1) != len(self.word2):
            raise _validation_error(
                "nonlinear_code.length_mismatch", "words must have equal length"
            )
        return self


class WordDistanceResult(StrictModel):
    """Exact Hamming relation between two binary words."""

    word1: BinaryWord
    word2: BinaryWord
    distance: StrictNonnegativeInt
    differing_coordinates: tuple[Coordinate, ...]
    weight1: StrictNonnegativeInt
    weight2: StrictNonnegativeInt
    support_intersection: StrictNonnegativeInt

    @model_validator(mode="after")
    def require_structural_words(self) -> Self:
        if not self.word1 or len(self.word1) != len(self.word2):
            raise _validation_error(
                "nonlinear_code.invalid_words",
                "result words must be nonempty and have equal length",
            )
        if self.distance > len(self.word1):
            raise _validation_error(
                "nonlinear_code.distance_bound",
                "distance cannot exceed the common word length",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        word1: BinaryWord,
        word2: BinaryWord,
        distance: int,
        differing_coordinates: tuple[int, ...],
        weight1: int,
        weight2: int,
        support_intersection: int,
    ) -> Self:
        return cls.model_construct(
            word1=word1,
            word2=word2,
            distance=distance,
            differing_coordinates=differing_coordinates,
            weight1=weight1,
            weight2=weight2,
            support_intersection=support_intersection,
        )


class BinaryCodeDistanceWitness(StrictModel):
    """One source-indexed word pair and its complete Hamming metadata."""

    left_index: StrictNonnegativeInt
    right_index: StrictNonnegativeInt
    left_word: BinaryWord
    right_word: BinaryWord
    left_support: tuple[Coordinate, ...]
    right_support: tuple[Coordinate, ...]
    differing_coordinates: tuple[Coordinate, ...]
    left_weight: StrictNonnegativeInt
    right_weight: StrictNonnegativeInt
    support_intersection: StrictNonnegativeInt
    distance: StrictNonnegativeInt


class ExplicitProfileRequest(StrictModel):
    """Compute the complete compact profile of one explicit binary code."""

    code: ExplicitBinaryCode = Field(
        description=(
            "Canonical explicit binary code. Admission derives C(M,2), "
            "pair-by-bitset-chunk work, dense histogram and witness size, and "
            "the retained-source result bound before pair enumeration."
        )
    )


class ExplicitProfileResult(StrictModel):
    """Complete pair and weight profile bound to its retained source."""

    source: ExplicitBinaryCode
    length: ExplicitLength
    cardinality: StrictNonnegativeInt
    pair_count: Annotated[int, Field(strict=True, ge=0, le=MAX_PROFILE_PAIRS)]
    weight_distribution: tuple[StrictNonnegativeInt, ...] = Field(
        max_length=MAX_EXPLICIT_CODE_LENGTH + 1
    )
    minimum_distance: StrictNonnegativeInt | None
    maximum_distance: StrictNonnegativeInt | None
    distance_histogram: tuple[StrictNonnegativeInt, ...] = Field(
        max_length=MAX_EXPLICIT_CODE_LENGTH + 1
    )
    minimum_distance_witness: BinaryCodeDistanceWitness | None
    maximum_distance_witness: BinaryCodeDistanceWitness | None

    @model_validator(mode="after")
    def require_structural_profile_relations(self) -> Self:
        if self.length != self.source.length:
            raise _validation_error(
                "nonlinear_code.length_mismatch",
                "length must equal the retained source length",
            )
        if self.cardinality != len(self.source.codewords):
            raise _validation_error(
                "nonlinear_code.cardinality_mismatch",
                "cardinality must equal the retained source cardinality",
            )
        if self.pair_count != self.cardinality * (self.cardinality - 1) // 2:
            raise _validation_error(
                "nonlinear_code.pair_count_mismatch",
                "pair_count must equal cardinality*(cardinality-1)/2",
            )
        if len(self.weight_distribution) != self.length + 1:
            raise _validation_error(
                "nonlinear_code.histogram_length",
                "weight_distribution must cover every possible weight",
            )
        if len(self.distance_histogram) != self.length + 1:
            raise _validation_error(
                "nonlinear_code.histogram_length",
                "distance_histogram must cover every possible distance",
            )
        if sum(self.weight_distribution) != self.cardinality:
            raise _validation_error(
                "nonlinear_code.weight_total_mismatch",
                "weight_distribution must total cardinality",
            )
        if sum(self.distance_histogram) != self.pair_count:
            raise _validation_error(
                "nonlinear_code.distance_total_mismatch",
                "distance_histogram must total pair_count",
            )
        if (self.minimum_distance is None) != (self.maximum_distance is None):
            raise _validation_error(
                "nonlinear_code.extremum_presence",
                "minimum and maximum distance must both be null or both be present",
            )
        return self

    @classmethod
    def _from_kernel(cls, **kwargs: Any) -> Self:
        return cls.model_construct(**kwargs)


class ConstantWeightProfileRequest(StrictModel):
    """Profile a nonempty constant-weight canonical explicit binary code."""

    code: ExplicitBinaryCode = Field(
        description=(
            "Nonempty canonical explicit binary code whose words all have the "
            "same Hamming weight; profile work and exact result size are "
            "derived before pair enumeration."
        )
    )


class ConstantWeightProfileResult(StrictModel):
    """Distance/intersection profile of a retained constant-weight code."""

    source: ExplicitBinaryCode
    length: ExplicitLength
    weight: StrictNonnegativeInt
    cardinality: StrictPositiveInt
    pair_count: Annotated[int, Field(strict=True, ge=0, le=MAX_PROFILE_PAIRS)]
    minimum_distance: StrictNonnegativeInt | None
    maximum_distance: StrictNonnegativeInt | None
    distance_histogram: tuple[StrictNonnegativeInt, ...] = Field(
        max_length=MAX_EXPLICIT_CODE_LENGTH + 1
    )
    intersection_histogram: tuple[StrictNonnegativeInt, ...] = Field(
        max_length=MAX_EXPLICIT_CODE_LENGTH + 1
    )
    minimum_distance_witness: BinaryCodeDistanceWitness | None
    maximum_distance_witness: BinaryCodeDistanceWitness | None

    @model_validator(mode="after")
    def require_structural_profile_relations(self) -> Self:
        if self.length != self.source.length:
            raise _validation_error(
                "nonlinear_code.length_mismatch",
                "length must equal the retained source length",
            )
        if self.cardinality != len(self.source.codewords):
            raise _validation_error(
                "nonlinear_code.cardinality_mismatch",
                "cardinality must equal the retained source cardinality",
            )
        if self.pair_count != self.cardinality * (self.cardinality - 1) // 2:
            raise _validation_error(
                "nonlinear_code.pair_count_mismatch",
                "pair_count must equal cardinality*(cardinality-1)/2",
            )
        if len(self.distance_histogram) != self.length + 1:
            raise _validation_error(
                "nonlinear_code.histogram_length",
                "distance_histogram must cover every possible distance",
            )
        if len(self.intersection_histogram) != self.length + 1:
            raise _validation_error(
                "nonlinear_code.histogram_length",
                "intersection_histogram must cover every possible intersection",
            )
        if (
            sum(self.distance_histogram) != self.pair_count
            or sum(self.intersection_histogram) != self.pair_count
        ):
            raise _validation_error(
                "nonlinear_code.histogram_total_mismatch",
                "both pair histograms must total pair_count",
            )
        if (self.minimum_distance is None) != (self.maximum_distance is None):
            raise _validation_error(
                "nonlinear_code.extremum_presence",
                "minimum and maximum distance must both be null or both be present",
            )
        return self

    @classmethod
    def _from_kernel(cls, **kwargs: Any) -> Self:
        return cls.model_construct(**kwargs)


class ToSetSystemRequest(StrictModel):
    """Map one canonical explicit code to its coordinate supports."""

    code: ExplicitBinaryCode


class ToSetSystemResult(StrictModel):
    """Exact source-indexed support blocks under the declared coordinate axis."""

    source: ExplicitBinaryCode
    length: ExplicitLength
    cardinality: StrictNonnegativeInt
    coordinate_axis: tuple[Coordinate, ...] = Field(max_length=MAX_EXPLICIT_CODE_LENGTH)
    supports: tuple[tuple[Coordinate, ...], ...]

    @model_validator(mode="after")
    def require_structural_support_relations(self) -> Self:
        if self.length != self.source.length:
            raise _validation_error(
                "nonlinear_code.length_mismatch",
                "length must equal the retained source length",
            )
        if self.cardinality != len(self.source.codewords):
            raise _validation_error(
                "nonlinear_code.cardinality_mismatch",
                "cardinality must equal the retained source cardinality",
            )
        if len(self.coordinate_axis) != self.length:
            raise _validation_error(
                "nonlinear_code.coordinate_axis_length",
                "coordinate_axis must have one coordinate per source position",
            )
        if len(self.supports) != self.cardinality:
            raise _validation_error(
                "nonlinear_code.support_count_mismatch",
                "supports must have one block per source word",
            )
        return self

    @classmethod
    def _from_kernel(cls, **kwargs: Any) -> Self:
        return cls.model_construct(**kwargs)


__all__ = [
    "BinaryCodeDistanceWitness",
    "ConstantWeightProfileRequest",
    "ConstantWeightProfileResult",
    "ConstantWeightRequest",
    "ConstantWeightResult",
    "ExplicitProfileRequest",
    "ExplicitProfileResult",
    "ToSetSystemRequest",
    "ToSetSystemResult",
    "WordDistanceRequest",
    "WordDistanceResult",
]
