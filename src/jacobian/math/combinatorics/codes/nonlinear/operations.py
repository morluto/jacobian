"""Exact native-integer kernels for nonlinear binary codes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.nonlinear._budget import (
    require_constant_weight_admission,
    require_profile_admission,
    require_word_distance_output_bound,
)
from jacobian.math.combinatorics.codes.nonlinear._models import (
    BinaryCodeDistanceWitness,
    ConstantWeightProfileResult,
    ConstantWeightResult,
    ExplicitProfileResult,
    ToSetSystemResult,
    WordDistanceResult,
)
from jacobian.math.combinatorics.codes.nonlinear.values import (
    BinaryWord,
    ExplicitBinaryCode,
)


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


def _admit[T](
    admission: Callable[[], T],
    *,
    location: tuple[str | int, ...],
    code: str,
) -> T:
    try:
        return admission()
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=str(exc),
        ) from exc


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


def _constant_weight(code: ExplicitBinaryCode) -> int:
    if not code.codewords:
        raise ValueError("constant-weight profile requires at least one codeword")
    weight = sum(code.codewords[0])
    if any(sum(word) != weight for word in code.codewords):
        raise ValueError("all codewords must have the same Hamming weight")
    return weight


def constant_weight_code(length: int, weight: int) -> ConstantWeightResult:
    """Generate the complete constant-weight binary code."""
    try:
        require_constant_weight_admission(length, weight)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("length", "weight"),
            code="nonlinear_code.constant_weight_not_admitted",
            message=str(exc),
        ) from exc
    code = _constant_weight_code(length, weight)
    return ConstantWeightResult._from_kernel(length=length, weight=weight, code=code)


def word_distance(word1: BinaryWord, word2: BinaryWord) -> WordDistanceResult:
    """Compute the exact Hamming relation between two words."""
    _admit(
        lambda: require_word_distance_output_bound(word1, word2),
        location=("word1", "word2"),
        code="nonlinear_code.word_distance_not_admitted",
    )
    distance, differing, weight1, weight2, intersection = _word_distance_data(
        word1, word2
    )
    return WordDistanceResult._from_kernel(
        word1=word1,
        word2=word2,
        distance=distance,
        differing_coordinates=differing,
        weight1=weight1,
        weight2=weight2,
        support_intersection=intersection,
    )


def explicit_profile(code: ExplicitBinaryCode) -> ExplicitProfileResult:
    """Compute the complete compact profile of an explicit binary code."""
    plan = _admit(
        lambda: require_profile_admission(code),
        location=("code",),
        code="nonlinear_code.profile_not_admitted",
    )
    profile = _explicit_profile_data(code)
    return ExplicitProfileResult._from_kernel(
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


def constant_weight_profile(code: ExplicitBinaryCode) -> ConstantWeightProfileResult:
    """Compute distance and intersection profiles of a constant-weight code."""
    plan = _admit(
        lambda: require_profile_admission(code),
        location=("code",),
        code="nonlinear_code.profile_not_admitted",
    )
    try:
        weight = _constant_weight(code)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("code",),
            code="nonlinear_code.not_constant_weight",
            message=str(exc),
        ) from exc
    profile = _constant_weight_profile_data(code)
    return ConstantWeightProfileResult._from_kernel(
        source=code,
        length=code.length,
        weight=weight,
        cardinality=len(code.codewords),
        pair_count=plan.pair_count,
        minimum_distance=profile.minimum_distance,
        maximum_distance=profile.maximum_distance,
        distance_histogram=profile.distance_histogram,
        intersection_histogram=profile.intersection_histogram,
        minimum_distance_witness=profile.minimum_distance_witness,
        maximum_distance_witness=profile.maximum_distance_witness,
    )


def to_set_system(code: ExplicitBinaryCode) -> ToSetSystemResult:
    """Native support conversion for one canonical explicit binary code."""
    return ToSetSystemResult._from_kernel(
        source=code,
        length=code.length,
        cardinality=len(code.codewords),
        coordinate_axis=tuple(range(code.length)),
        supports=tuple(
            tuple(i for i, bit in enumerate(word) if bit) for word in code.codewords
        ),
    )


__all__ = [
    "constant_weight_code",
    "constant_weight_profile",
    "explicit_profile",
    "to_set_system",
    "word_distance",
]
