"""Typed wire contracts for nonlinear binary code operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.code_nonlinear._budget import (
    MAX_CODE_RESULT_BYTES,
    MAX_PROFILE_PAIRS,
    distance_profile_wire_upper_bound,
    require_constant_weight_admission,
    require_pair_work_admission,
    require_profile_admission,
    require_set_system_output_bound,
    require_word_distance_output_bound,
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


def _require_admission(action: Callable[[], Any], code: str) -> Any:
    try:
        return action()
    except ValueError as exc:
        raise _validation_error(code, str(exc)) from exc


def _require_result_bound(bound: int, label: str) -> None:
    if bound > MAX_CODE_RESULT_BYTES:
        raise _validation_error(
            "nonlinear_code.result_bound",
            f"{label} can use up to {bound} canonical JSON bytes, exceeding the "
            f"{MAX_CODE_RESULT_BYTES}-byte result bound",
        )


def _require_constant_weight(code: ExplicitBinaryCode) -> int:
    if not code.codewords:
        raise _validation_error(
            "nonlinear_code.constant_weight_empty",
            "constant-weight profile requires at least one codeword",
        )
    weight = sum(code.codewords[0])
    if any(sum(word) != weight for word in code.codewords):
        raise _validation_error(
            "nonlinear_code.constant_weight_mismatch",
            "all codewords must have the same Hamming weight",
        )
    return weight


class BinaryCodeRequest(StrictModel):
    """Request the legacy compact distance profile of one canonical code."""

    code: ExplicitBinaryCode

    @model_validator(mode="after")
    def require_bounded_profile(self) -> Self:
        _require_admission(
            lambda: require_pair_work_admission(self.code),
            "nonlinear_code.admission_bound",
        )
        _require_result_bound(
            distance_profile_wire_upper_bound(self.code), "distance profile result"
        )
        return self


class ConstantWeightRequest(StrictModel):
    """Generate all binary words of one bounded length and weight."""

    length: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]
    weight: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]

    @model_validator(mode="after")
    def require_valid_weight(self) -> Self:
        _require_admission(
            lambda: require_constant_weight_admission(self.length, self.weight),
            "nonlinear_code.admission_bound",
        )
        return self


class DistanceProfileResult(StrictModel):
    """Minimum pair distance and per-word weights, bound to the source code."""

    source: ExplicitBinaryCode
    minimum_distance: StrictNonnegativeInt | None
    weight_profile: tuple[StrictNonnegativeInt, ...]

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _distance_profile_data

        _require_admission(
            lambda: require_pair_work_admission(self.source),
            "nonlinear_code.admission_bound",
        )
        _require_result_bound(
            distance_profile_wire_upper_bound(self.source),
            "distance profile result",
        )
        minimum_distance, weights = _distance_profile_data(self.source)
        if self.minimum_distance != minimum_distance:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "minimum_distance must replay from the retained source",
            )
        if self.weight_profile != weights:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "weight_profile must replay from the retained source",
            )
        return self


class ConstantWeightResult(StrictModel):
    """Complete generated constant-weight code, bound to length and weight."""

    length: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]
    weight: Annotated[int, Field(strict=True, ge=0, le=MAX_EXPLICIT_CODE_LENGTH)]
    code: ExplicitBinaryCode
    count: StrictPositiveInt

    @model_validator(mode="after")
    def bind_generated_code(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _constant_weight_code

        _require_admission(
            lambda: require_constant_weight_admission(self.length, self.weight),
            "nonlinear_code.admission_bound",
        )
        expected = _constant_weight_code(self.length, self.weight)
        if self.code != expected:
            raise _validation_error(
                "nonlinear_code.generated_code_mismatch",
                "code must contain every word of the declared weight",
            )
        if self.count != len(expected.codewords):
            raise _validation_error(
                "nonlinear_code.cardinality_mismatch",
                "count must equal the generated code cardinality",
            )
        return self


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
        _require_admission(
            lambda: require_word_distance_output_bound(self.word1, self.word2),
            "nonlinear_code.admission_bound",
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
    def bind_distance(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _word_distance_data

        if not self.word1 or len(self.word1) != len(self.word2):
            raise _validation_error(
                "nonlinear_code.invalid_words",
                "result words must be nonempty and have equal length",
            )
        _require_admission(
            lambda: require_word_distance_output_bound(self.word1, self.word2),
            "nonlinear_code.admission_bound",
        )
        expected = _word_distance_data(self.word1, self.word2)
        if (
            self.distance,
            self.differing_coordinates,
            self.weight1,
            self.weight2,
            self.support_intersection,
        ) != expected:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "word-distance fields must replay from the retained words",
            )
        return self


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


def _require_extremal_witness(
    source: ExplicitBinaryCode,
    witness: BinaryCodeDistanceWitness | None,
    expected_distance: int | None,
    label: str,
) -> None:
    from jacobian.math.code_nonlinear._operations import _word_distance_data

    if expected_distance is None:
        if witness is not None:
            raise _validation_error(
                "nonlinear_code.witness_unexpected",
                f"{label} witness must be null when there are no pairs",
            )
        return
    if witness is None:
        raise _validation_error(
            "nonlinear_code.witness_missing",
            f"{label} witness is required when pairs exist",
        )
    cardinality = len(source.codewords)
    if not 0 <= witness.left_index < witness.right_index < cardinality:
        raise _validation_error(
            "nonlinear_code.witness_indices",
            f"{label} witness indices must name one unordered source pair",
        )
    left_word = source.codewords[witness.left_index]
    right_word = source.codewords[witness.right_index]
    if witness.left_word != left_word or witness.right_word != right_word:
        raise _validation_error(
            "nonlinear_code.witness_source_mismatch",
            f"{label} witness words must match their source indices",
        )
    distance, differing, left_weight, right_weight, intersection = _word_distance_data(
        left_word, right_word
    )
    left_support = tuple(i for i, bit in enumerate(left_word) if bit)
    right_support = tuple(i for i, bit in enumerate(right_word) if bit)
    if (
        witness.left_support,
        witness.right_support,
        witness.differing_coordinates,
        witness.left_weight,
        witness.right_weight,
        witness.support_intersection,
        witness.distance,
    ) != (
        left_support,
        right_support,
        differing,
        left_weight,
        right_weight,
        intersection,
        distance,
    ):
        raise _validation_error(
            "nonlinear_code.witness_replay_mismatch",
            f"{label} witness metadata must replay from its source pair",
        )
    if distance != expected_distance:
        raise _validation_error(
            "nonlinear_code.witness_distance_mismatch",
            f"{label} witness must attain the declared extremal distance",
        )


class ExplicitProfileRequest(StrictModel):
    """Compute the complete compact profile of one explicit binary code."""

    code: ExplicitBinaryCode = Field(
        description=(
            "Canonical explicit binary code. Admission derives C(M,2), "
            "pair-by-bitset-chunk work, dense histogram and witness size, and "
            "the retained-source result bound before pair enumeration."
        )
    )

    @model_validator(mode="after")
    def require_bounded_profile(self) -> Self:
        _require_admission(
            lambda: require_profile_admission(self.code),
            "nonlinear_code.admission_bound",
        )
        return self


class ExplicitProfileResult(StrictModel):
    """Complete pair and weight profile, replayable from its retained source."""

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
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import _explicit_profile_data

        plan = _require_admission(
            lambda: require_profile_admission(self.source),
            "nonlinear_code.admission_bound",
        )
        expected = _explicit_profile_data(self.source)
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
        if self.pair_count != plan.pair_count:
            raise _validation_error(
                "nonlinear_code.pair_count_mismatch",
                "pair_count must equal cardinality*(cardinality-1)/2",
            )
        if self.weight_distribution != expected.weight_distribution:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "weight_distribution must replay from the retained source",
            )
        if self.minimum_distance != expected.minimum_distance:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "minimum_distance must replay from the retained source",
            )
        if self.maximum_distance != expected.maximum_distance:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "maximum_distance must replay from the retained source",
            )
        if self.distance_histogram != expected.distance_histogram:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "distance_histogram must replay from the retained source",
            )
        _require_extremal_witness(
            self.source,
            self.minimum_distance_witness,
            self.minimum_distance,
            "minimum-distance",
        )
        _require_extremal_witness(
            self.source,
            self.maximum_distance_witness,
            self.maximum_distance,
            "maximum-distance",
        )
        return self


class ConstantWeightProfileRequest(StrictModel):
    """Profile a nonempty constant-weight canonical explicit binary code."""

    code: ExplicitBinaryCode = Field(
        description=(
            "Nonempty canonical explicit binary code whose words all have the "
            "same Hamming weight; profile work and exact result size are "
            "derived before pair enumeration."
        )
    )

    @model_validator(mode="after")
    def require_valid_constant_weight(self) -> Self:
        _require_constant_weight(self.code)
        _require_admission(
            lambda: require_profile_admission(self.code),
            "nonlinear_code.admission_bound",
        )
        return self


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
    def bind_profile(self) -> Self:
        from jacobian.math.code_nonlinear._operations import (
            _constant_weight_profile_data,
        )

        weight = _require_constant_weight(self.source)
        plan = _require_admission(
            lambda: require_profile_admission(self.source),
            "nonlinear_code.admission_bound",
        )
        expected = _constant_weight_profile_data(self.source)
        if self.length != self.source.length:
            raise _validation_error(
                "nonlinear_code.length_mismatch",
                "length must equal the retained source length",
            )
        if self.weight != weight:
            raise _validation_error(
                "nonlinear_code.weight_mismatch",
                "weight must equal every retained source word weight",
            )
        if self.cardinality != len(self.source.codewords):
            raise _validation_error(
                "nonlinear_code.cardinality_mismatch",
                "cardinality must equal the retained source cardinality",
            )
        if self.pair_count != plan.pair_count:
            raise _validation_error(
                "nonlinear_code.pair_count_mismatch",
                "pair_count must equal cardinality*(cardinality-1)/2",
            )
        if self.minimum_distance != expected.minimum_distance:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "minimum_distance must replay from the retained source",
            )
        if self.maximum_distance != expected.maximum_distance:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "maximum_distance must replay from the retained source",
            )
        if self.distance_histogram != expected.distance_histogram:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "distance_histogram must replay from the retained source",
            )
        if self.intersection_histogram != expected.intersection_histogram:
            raise _validation_error(
                "nonlinear_code.replay_mismatch",
                "intersection_histogram must replay from the retained source",
            )
        _require_extremal_witness(
            self.source,
            self.minimum_distance_witness,
            self.minimum_distance,
            "minimum-distance",
        )
        _require_extremal_witness(
            self.source,
            self.maximum_distance_witness,
            self.maximum_distance,
            "maximum-distance",
        )
        return self


class ToSetSystemRequest(StrictModel):
    """Map one canonical explicit code to its coordinate supports."""

    code: ExplicitBinaryCode

    @model_validator(mode="after")
    def require_bounded_result(self) -> Self:
        _require_admission(
            lambda: require_set_system_output_bound(self.code),
            "nonlinear_code.admission_bound",
        )
        return self


class ToSetSystemResult(StrictModel):
    """Exact source-indexed support blocks under the declared coordinate axis."""

    source: ExplicitBinaryCode
    length: ExplicitLength
    cardinality: StrictNonnegativeInt
    coordinate_axis: tuple[Coordinate, ...] = Field(max_length=MAX_EXPLICIT_CODE_LENGTH)
    supports: tuple[tuple[Coordinate, ...], ...]

    @model_validator(mode="after")
    def bind_supports(self) -> Self:
        expected_axis = tuple(range(self.source.length))
        expected_supports = tuple(
            tuple(i for i, bit in enumerate(word) if bit)
            for word in self.source.codewords
        )
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
        if self.coordinate_axis != expected_axis:
            raise _validation_error(
                "nonlinear_code.coordinate_axis_mismatch",
                "coordinate_axis must be exactly 0 through length-1",
            )
        if self.supports != expected_supports:
            raise _validation_error(
                "nonlinear_code.supports_mismatch",
                "supports must be the exact 1-positions of source words",
            )
        return self


__all__ = [
    "BinaryCodeDistanceWitness",
    "BinaryCodeRequest",
    "ConstantWeightProfileRequest",
    "ConstantWeightProfileResult",
    "ConstantWeightRequest",
    "ConstantWeightResult",
    "DistanceProfileResult",
    "ExplicitProfileRequest",
    "ExplicitProfileResult",
    "ToSetSystemRequest",
    "ToSetSystemResult",
    "WordDistanceRequest",
    "WordDistanceResult",
]
