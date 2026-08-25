"""Pre-execution work, intermediate, and exact-output admission."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.math.code_nonlinear.values import (
    MAX_EXPLICIT_CODE_BITS,
    BinaryWord,
    ExplicitBinaryCode,
)

# A normal operation invocation visits every unordered pair once in the kernel
# and once more when result construction replays the source-bound conclusion.
# The standard A(23,6,10) construction has length 23, minimum distance 6,
# constant weight 10, and 2992 words, so it uses ``2 * 4,474,536``
# pair-by-chunk units.
MAX_PROFILE_PAIRS = 5_000_000
PROFILE_PAIR_PASSES = 2
MAX_PROFILE_BITSET_CHUNK_WORK = 10_000_000
BITSET_CHUNK_BITS = 30

# Retained sources, dense histograms, and the two compact witnesses must fit
# below the canonical transport's 10 MiB ceiling.  Four MiB leaves room for
# the operation envelope and is checked from the source before pair replay.
MAX_CODE_RESULT_BYTES = 4 * 1024 * 1024

# Enumerating every ``length``-bit word of weight ``w`` visits exactly
# ``binomial(length, w)`` candidates and materializes exactly
# ``length * binomial(length, w)`` binary entries, so that one quantity
# bounds the combinations work, the intermediate codeword list, and the
# exact generated result.  Capping it at the canonical explicit-code source
# bound admits materially larger useful cases than any fixed length cap;
# the documented A(23,6,10) construction needs 68,816 entries.
MAX_GENERATED_CODE_ENTRIES = MAX_EXPLICIT_CODE_BITS


@dataclass(frozen=True, slots=True)
class ProfileAdmission:
    pair_count: int
    pair_passes: int
    bitset_chunks: int
    bitset_chunk_work: int
    result_wire_upper_bound: int


def _digits(value: int) -> int:
    return len(str(value))


def _binary_word_wire_bytes(length: int) -> int:
    # ``[0,1,...]`` has one byte per bit, one comma between bits, and brackets.
    return max(2, 2 * length + 1)


def _array_wire_bytes(length: int, maximum_value: int) -> int:
    if length == 0:
        return 2
    return 1 + length * (_digits(maximum_value) + 1)


def _sum_coordinate_digits(length: int) -> int:
    """Return the exact decimal digits in ``0, ..., length-1``."""
    if length == 0:
        return 0
    total = 1  # coordinate 0
    lower = 1
    digits = 1
    maximum = length - 1
    while lower <= maximum:
        upper = min(maximum, lower * 10 - 1)
        total += (upper - lower + 1) * digits
        lower *= 10
        digits += 1
    return total


def _coordinate_axis_wire_bytes(length: int) -> int:
    if length == 0:
        return 2
    return 2 + _sum_coordinate_digits(length) + length - 1


def _support_wire_bytes(word: BinaryWord) -> int:
    support_size = 0
    coordinate_digits = 0
    for coordinate, bit in enumerate(word):
        if bit:
            support_size += 1
            coordinate_digits += _digits(coordinate)
    return 2 + coordinate_digits + max(0, support_size - 1)


def source_wire_upper_bound(code: ExplicitBinaryCode) -> int:
    cardinality = len(code.codewords)
    word_bytes = _binary_word_wire_bytes(code.length)
    words_bytes = 2 if cardinality == 0 else 1 + cardinality * (word_bytes + 1)
    # Fixed object keys, braces, comma, and scalar punctuation fit well below
    # this 64-byte allowance.
    return 64 + _digits(code.length) + words_bytes


def _profile_result_wire_upper_bound(code: ExplicitBinaryCode, pair_count: int) -> int:
    length = code.length
    cardinality = len(code.codewords)
    histogram_count_digits = max(_digits(cardinality), _digits(pair_count))
    histogram_bytes = 1 + (length + 1) * (histogram_count_digits + 1)

    witness_bytes = 0
    if pair_count:
        largest_supports = sorted(
            (_support_wire_bytes(word) for word in code.codewords), reverse=True
        )[:2]
        largest_supports += [2] * (2 - len(largest_supports))

        # A witness retains both actual words, both supports, every differing
        # coordinate, indices, weights, intersection, and distance.  A profile
        # with pairs returns two witnesses; zero-pair profiles return none.
        witness_bytes = (
            2 * _binary_word_wire_bytes(length)
            + sum(largest_supports)
            + _coordinate_axis_wire_bytes(length)
            + 512
            + 10 * _digits(max(length, cardinality, pair_count, 1))
        )
    # Three dense histograms conservatively cover the explicit weight/distance
    # result and the constant-weight distance/intersection result.
    return (
        source_wire_upper_bound(code) + 3 * histogram_bytes + 2 * witness_bytes + 2_048
    )


def require_profile_admission(code: ExplicitBinaryCode) -> ProfileAdmission:
    pair_count, bitset_chunks, bitset_chunk_work = require_pair_work_admission(code)
    result_wire_upper_bound = _profile_result_wire_upper_bound(code, pair_count)
    if result_wire_upper_bound > MAX_CODE_RESULT_BYTES:
        raise ValueError(
            "explicit profile retained source, histograms, and witnesses can use "
            f"up to {result_wire_upper_bound} canonical JSON bytes, exceeding the "
            f"{MAX_CODE_RESULT_BYTES}-byte result bound"
        )
    return ProfileAdmission(
        pair_count=pair_count,
        pair_passes=PROFILE_PAIR_PASSES,
        bitset_chunks=bitset_chunks,
        bitset_chunk_work=bitset_chunk_work,
        result_wire_upper_bound=result_wire_upper_bound,
    )


def require_pair_work_admission(code: ExplicitBinaryCode) -> tuple[int, int, int]:
    """Admit the kernel and source-replay passes before allocating the result."""
    cardinality = len(code.codewords)
    pair_count = cardinality * (cardinality - 1) // 2
    if pair_count > MAX_PROFILE_PAIRS:
        raise ValueError(
            f"explicit profile has {pair_count} unordered pairs, exceeding the "
            f"{MAX_PROFILE_PAIRS}-pair bound"
        )
    bitset_chunks = (code.length + BITSET_CHUNK_BITS - 1) // BITSET_CHUNK_BITS
    bitset_chunk_work = pair_count * bitset_chunks * PROFILE_PAIR_PASSES
    if bitset_chunk_work > MAX_PROFILE_BITSET_CHUNK_WORK:
        raise ValueError(
            "explicit profile requires "
            f"{bitset_chunk_work} pair-by-bitset-chunk units "
            f"({pair_count} pairs * {bitset_chunks} chunks * "
            f"{PROFILE_PAIR_PASSES} kernel/replay passes), exceeding the "
            f"{MAX_PROFILE_BITSET_CHUNK_WORK}-unit bound"
        )
    return pair_count, bitset_chunks, bitset_chunk_work


def require_set_system_output_bound(code: ExplicitBinaryCode) -> int:
    cardinality = len(code.codewords)
    supports = tuple(_support_wire_bytes(word) for word in code.codewords)
    supports_bytes = 2 if cardinality == 0 else 1 + sum(v + 1 for v in supports)
    bound = (
        source_wire_upper_bound(code)
        + _coordinate_axis_wire_bytes(code.length)
        + supports_bytes
        + 1_024
    )
    if bound > MAX_CODE_RESULT_BYTES:
        raise ValueError(
            "support conversion retained source, coordinate axis, and blocks can "
            f"use up to {bound} canonical JSON bytes, exceeding the "
            f"{MAX_CODE_RESULT_BYTES}-byte result bound"
        )
    return bound


def _differing_coordinate_wire_bytes(word1: BinaryWord, word2: BinaryWord) -> int:
    coordinate_digits = 0
    differing_count = 0
    for coordinate, (left, right) in enumerate(zip(word1, word2, strict=True)):
        if left != right:
            differing_count += 1
            coordinate_digits += _digits(coordinate)
    return 2 + coordinate_digits + max(0, differing_count - 1)


def require_word_distance_output_bound(word1: BinaryWord, word2: BinaryWord) -> int:
    differing_bytes = _differing_coordinate_wire_bytes(word1, word2)
    bound = 2 * _binary_word_wire_bytes(len(word1)) + differing_bytes + 1_024
    if bound > MAX_CODE_RESULT_BYTES:
        raise ValueError(
            "word-distance result can use up to "
            f"{bound} canonical JSON bytes including {differing_bytes} bytes of "
            f"differing coordinates, exceeding the "
            f"{MAX_CODE_RESULT_BYTES}-byte result bound"
        )
    return bound


def constant_weight_result_wire_upper_bound(length: int, cardinality: int) -> int:
    word_bytes = _binary_word_wire_bytes(length)
    words_bytes = 2 if cardinality == 0 else 1 + cardinality * (word_bytes + 1)
    # Fixed object keys, braces, commas, and scalar punctuation fit well
    # below this 64-byte allowance.
    return 64 + words_bytes + 3 * (_digits(length) + _digits(cardinality))


def _binomial_within_entry_budget(length: int, weight: int) -> int:
    # Iterating the multiplicative identity stops at the first step where
    # ``length * cardinality`` exceeds the entry bound.  After symmetry
    # reduction every step multiplies by at least two, so a rejected request
    # stops within logarithmically many steps instead of computing the full
    # central coefficient, and every reported count stays small enough to
    # interpolate under Python's integer-to-decimal digit limit.
    reduced_weight = min(weight, length - weight)
    cardinality = 1
    for step in range(reduced_weight + 1):
        if step:
            cardinality = cardinality * (length - reduced_weight + step) // step
        entries = length * cardinality
        if entries > MAX_GENERATED_CODE_ENTRIES:
            raise ValueError(
                "constant-weight generation materializes "
                f"{entries} entries ({length} coordinates * {cardinality} words), "
                f"exceeding the {MAX_GENERATED_CODE_ENTRIES}-entry bound"
            )
    return cardinality


def require_constant_weight_admission(length: int, weight: int) -> int:
    """Admit the complete weight-``weight`` generation before enumerating."""
    if not 0 <= weight <= length:
        raise ValueError("weight cannot exceed length")
    cardinality = _binomial_within_entry_budget(length, weight)
    wire_bytes = constant_weight_result_wire_upper_bound(length, cardinality)
    if wire_bytes > MAX_CODE_RESULT_BYTES:
        raise ValueError(
            "constant-weight generation can use up to "
            f"{wire_bytes} canonical JSON bytes, exceeding the "
            f"{MAX_CODE_RESULT_BYTES}-byte result bound"
        )
    return cardinality


__all__ = [
    "BITSET_CHUNK_BITS",
    "MAX_CODE_RESULT_BYTES",
    "MAX_GENERATED_CODE_ENTRIES",
    "MAX_PROFILE_BITSET_CHUNK_WORK",
    "MAX_PROFILE_PAIRS",
    "PROFILE_PAIR_PASSES",
    "ProfileAdmission",
    "constant_weight_result_wire_upper_bound",
    "require_constant_weight_admission",
    "require_pair_work_admission",
    "require_profile_admission",
    "require_set_system_output_bound",
    "require_word_distance_output_bound",
    "source_wire_upper_bound",
]
