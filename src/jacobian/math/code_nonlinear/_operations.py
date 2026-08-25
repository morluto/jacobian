"""Exact native-integer kernels for nonlinear binary codes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from jacobian.math.code_nonlinear._budget import require_profile_admission
from jacobian.math.code_nonlinear._models import (
    BinaryCodeDistanceWitness,
    ConstantWeightProfileRequest,
    ConstantWeightProfileResult,
    ConstantWeightRequest,
    ConstantWeightResult,
    ExplicitProfileRequest,
    ExplicitProfileResult,
    ToSetSystemRequest,
    ToSetSystemResult,
    WordDistanceRequest,
    WordDistanceResult,
)
from jacobian.math.code_nonlinear.values import BinaryWord, ExplicitBinaryCode


@dataclass(frozen=True, slots=True)
class _ExplicitProfileData:
    weight_distribution: tuple[int, ...]
    minimum_distance: int | None
    maximum_distance: int | None
    distance_histogram: tuple[int, ...]
    minimum_distance_witness: BinaryCodeDistanceWitness | None
    maximum_distance_witness: BinaryCodeDistanceWitness | None


@dataclass(frozen=True, slots=True)
class _ConstantWeightProfileData:
    minimum_distance: int | None
    maximum_distance: int | None
    distance_histogram: tuple[int, ...]
    intersection_histogram: tuple[int, ...]
    minimum_distance_witness: BinaryCodeDistanceWitness | None
    maximum_distance_witness: BinaryCodeDistanceWitness | None


def _word_to_bitset(word: BinaryWord) -> int:
    """Pack coordinate ``i`` into bit ``i`` in one native Python integer."""
    packed = bytearray((len(word) + 7) // 8)
    for coordinate, bit in enumerate(word):
        if bit:
            packed[coordinate // 8] |= 1 << (coordinate % 8)
    return int.from_bytes(packed, "little")


def _word_distance_data(
    word1: BinaryWord, word2: BinaryWord
) -> tuple[int, tuple[int, ...], int, int, int]:
    left = _word_to_bitset(word1)
    right = _word_to_bitset(word2)
    differing = tuple(
        coordinate
        for coordinate, (left_bit, right_bit) in enumerate(
            zip(word1, word2, strict=True)
        )
        if left_bit != right_bit
    )
    return (
        (left ^ right).bit_count(),
        differing,
        left.bit_count(),
        right.bit_count(),
        (left & right).bit_count(),
    )


def _distance_witness(
    code: ExplicitBinaryCode, left_index: int, right_index: int
) -> BinaryCodeDistanceWitness:
    left_word = code.codewords[left_index]
    right_word = code.codewords[right_index]
    distance, differing, left_weight, right_weight, intersection = _word_distance_data(
        left_word, right_word
    )
    return BinaryCodeDistanceWitness(
        left_index=left_index,
        right_index=right_index,
        left_word=left_word,
        right_word=right_word,
        left_support=tuple(i for i, bit in enumerate(left_word) if bit),
        right_support=tuple(i for i, bit in enumerate(right_word) if bit),
        differing_coordinates=differing,
        left_weight=left_weight,
        right_weight=right_weight,
        support_intersection=intersection,
        distance=distance,
    )


def _explicit_profile_data(code: ExplicitBinaryCode) -> _ExplicitProfileData:
    require_profile_admission(code)
    bitsets = tuple(_word_to_bitset(word) for word in code.codewords)
    weight_distribution = [0] * (code.length + 1)
    for word in bitsets:
        weight_distribution[word.bit_count()] += 1

    distance_histogram = [0] * (code.length + 1)
    minimum: int | None = None
    maximum: int | None = None
    minimum_pair: tuple[int, int] | None = None
    maximum_pair: tuple[int, int] | None = None
    for left_index, left in enumerate(bitsets):
        for right_index in range(left_index + 1, len(bitsets)):
            distance = (left ^ bitsets[right_index]).bit_count()
            distance_histogram[distance] += 1
            if minimum is None or distance < minimum:
                minimum = distance
                minimum_pair = (left_index, right_index)
            if maximum is None or distance > maximum:
                maximum = distance
                maximum_pair = (left_index, right_index)

    return _ExplicitProfileData(
        weight_distribution=tuple(weight_distribution),
        minimum_distance=minimum,
        maximum_distance=maximum,
        distance_histogram=tuple(distance_histogram),
        minimum_distance_witness=(
            _distance_witness(code, *minimum_pair) if minimum_pair is not None else None
        ),
        maximum_distance_witness=(
            _distance_witness(code, *maximum_pair) if maximum_pair is not None else None
        ),
    )


def _constant_weight_profile_data(
    code: ExplicitBinaryCode,
) -> _ConstantWeightProfileData:
    require_profile_admission(code)
    bitsets = tuple(_word_to_bitset(word) for word in code.codewords)
    weight = bitsets[0].bit_count()
    distance_histogram = [0] * (code.length + 1)
    intersection_histogram = [0] * (code.length + 1)
    minimum: int | None = None
    maximum: int | None = None
    minimum_pair: tuple[int, int] | None = None
    maximum_pair: tuple[int, int] | None = None

    for left_index, left in enumerate(bitsets):
        for right_index in range(left_index + 1, len(bitsets)):
            right = bitsets[right_index]
            distance = (left ^ right).bit_count()
            intersection = (left & right).bit_count()
            if distance != 2 * (weight - intersection):
                raise AssertionError(
                    "constant-weight Hamming and support-intersection identities differ"
                )
            distance_histogram[distance] += 1
            intersection_histogram[intersection] += 1
            if minimum is None or distance < minimum:
                minimum = distance
                minimum_pair = (left_index, right_index)
            if maximum is None or distance > maximum:
                maximum = distance
                maximum_pair = (left_index, right_index)

    return _ConstantWeightProfileData(
        minimum_distance=minimum,
        maximum_distance=maximum,
        distance_histogram=tuple(distance_histogram),
        intersection_histogram=tuple(intersection_histogram),
        minimum_distance_witness=(
            _distance_witness(code, *minimum_pair) if minimum_pair is not None else None
        ),
        maximum_distance_witness=(
            _distance_witness(code, *maximum_pair) if maximum_pair is not None else None
        ),
    )


def _constant_weight_code(length: int, weight: int) -> ExplicitBinaryCode:
    codewords: list[tuple[int, ...]] = []
    for support in combinations(range(length), weight):
        support_set = set(support)
        codewords.append(tuple(1 if i in support_set else 0 for i in range(length)))
    return ExplicitBinaryCode(length=length, codewords=tuple(codewords))


def compute_constant_weight(request: ConstantWeightRequest) -> ConstantWeightResult:
    """Generate the complete constant-weight binary code."""
    code = _constant_weight_code(request.length, request.weight)
    return ConstantWeightResult(
        length=request.length,
        weight=request.weight,
        code=code,
        count=len(code.codewords),
    )


def compute_word_distance(request: WordDistanceRequest) -> WordDistanceResult:
    """Compute the exact Hamming relation between two words."""
    distance, differing, weight1, weight2, intersection = _word_distance_data(
        request.word1, request.word2
    )
    return WordDistanceResult(
        word1=request.word1,
        word2=request.word2,
        distance=distance,
        differing_coordinates=differing,
        weight1=weight1,
        weight2=weight2,
        support_intersection=intersection,
    )


def compute_explicit_profile(request: ExplicitProfileRequest) -> ExplicitProfileResult:
    """Compute the complete compact profile of an explicit binary code."""
    code = request.code
    plan = require_profile_admission(code)
    profile = _explicit_profile_data(code)
    return ExplicitProfileResult(
        source=code,
        length=code.length,
        cardinality=len(code.codewords),
        pair_count=plan.pair_count,
        weight_distribution=profile.weight_distribution,
        minimum_distance=profile.minimum_distance,
        maximum_distance=profile.maximum_distance,
        distance_histogram=profile.distance_histogram,
        minimum_distance_witness=profile.minimum_distance_witness,
        maximum_distance_witness=profile.maximum_distance_witness,
    )


def compute_constant_weight_profile(
    request: ConstantWeightProfileRequest,
) -> ConstantWeightProfileResult:
    """Compute distance and intersection profiles of a constant-weight code."""
    code = request.code
    plan = require_profile_admission(code)
    profile = _constant_weight_profile_data(code)
    return ConstantWeightProfileResult(
        source=code,
        length=code.length,
        weight=sum(code.codewords[0]),
        cardinality=len(code.codewords),
        pair_count=plan.pair_count,
        minimum_distance=profile.minimum_distance,
        maximum_distance=profile.maximum_distance,
        distance_histogram=profile.distance_histogram,
        intersection_histogram=profile.intersection_histogram,
        minimum_distance_witness=profile.minimum_distance_witness,
        maximum_distance_witness=profile.maximum_distance_witness,
    )


def compute_to_set_system(request: ToSetSystemRequest) -> ToSetSystemResult:
    """Map each source word to its coordinate support."""
    code = request.code
    return ToSetSystemResult(
        source=code,
        length=code.length,
        cardinality=len(code.codewords),
        coordinate_axis=tuple(range(code.length)),
        supports=tuple(
            tuple(i for i, bit in enumerate(word) if bit) for word in code.codewords
        ),
    )


def to_set_system(code: ExplicitBinaryCode) -> ToSetSystemResult:
    """Native support conversion for one canonical explicit binary code."""
    return ToSetSystemResult(
        source=code,
        length=code.length,
        cardinality=len(code.codewords),
        coordinate_axis=tuple(range(code.length)),
        supports=tuple(
            tuple(i for i, bit in enumerate(word) if bit) for word in code.codewords
        ),
    )


__all__ = [
    "compute_constant_weight",
    "compute_constant_weight_profile",
    "compute_explicit_profile",
    "compute_to_set_system",
    "compute_word_distance",
    "to_set_system",
]
