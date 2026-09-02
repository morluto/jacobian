"""Pre-execution work, intermediate, and exact-output admission."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.math.combinatorics.codes.nonlinear.values import (
    MAX_EXPLICIT_CODE_BITS,
    ExplicitBinaryCode,
)

# A normal operation invocation visits every unordered pair once in the kernel.
# The standard A(23,6,10) construction has length 23, minimum distance 6,
# constant weight 10, and 2992 words, so it uses ``4,474,536``
# pair-by-chunk units.
MAX_PROFILE_PAIRS = 5_000_000
PROFILE_PAIR_PASSES = 1
MAX_PROFILE_BITSET_CHUNK_WORK = 10_000_000
BITSET_CHUNK_BITS = 30

# Retained sources, dense histograms, and the two compact witnesses are bounded
# directly by codeword, histogram-cell, and witness cardinality.
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


def require_profile_admission(code: ExplicitBinaryCode) -> ProfileAdmission:
    pair_count, bitset_chunks, bitset_chunk_work = require_pair_work_admission(code)
    return ProfileAdmission(
        pair_count=pair_count,
        pair_passes=PROFILE_PAIR_PASSES,
        bitset_chunks=bitset_chunks,
        bitset_chunk_work=bitset_chunk_work,
    )


def require_pair_work_admission(code: ExplicitBinaryCode) -> tuple[int, int, int]:
    """Admit the kernel pair scan before allocating the result."""
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
            f"{PROFILE_PAIR_PASSES} kernel pass), exceeding the "
            f"{MAX_PROFILE_BITSET_CHUNK_WORK}-unit bound"
        )
    return pair_count, bitset_chunks, bitset_chunk_work


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
    return cardinality


__all__ = [
    "BITSET_CHUNK_BITS",
    "MAX_GENERATED_CODE_ENTRIES",
    "MAX_PROFILE_BITSET_CHUNK_WORK",
    "MAX_PROFILE_PAIRS",
    "PROFILE_PAIR_PASSES",
    "ProfileAdmission",
    "require_constant_weight_admission",
    "require_pair_work_admission",
    "require_profile_admission",
]
