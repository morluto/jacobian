"""Domain functions for nonlinear binary code operations."""

from __future__ import annotations

from itertools import combinations

from jacobian.math.code_nonlinear._models import (
    BinaryCodeRequest,
    ConstantWeightRequest,
    ConstantWeightResult,
    DistanceProfileResult,
)


def compute_distance_profile(request: BinaryCodeRequest) -> DistanceProfileResult:
    """Compute minimum Hamming distance and weight profile of a binary code."""
    codewords = request.codewords
    if len(codewords) == 1:
        w = sum(codewords[0])
        return DistanceProfileResult(
            minimum_distance=w, weight_profile=(w,), method="EXACT_ENUMERATION"
        )
    min_dist = len(codewords[0]) + 1
    for i, w1 in enumerate(codewords):
        for w2 in codewords[i + 1 :]:
            dist = sum(a != b for a, b in zip(w1, w2, strict=True))
            if dist < min_dist:
                min_dist = dist
    return DistanceProfileResult(
        minimum_distance=min_dist,
        weight_profile=tuple(sum(w) for w in codewords),
    )


def compute_constant_weight(request: ConstantWeightRequest) -> ConstantWeightResult:
    """Generate all constant-weight binary words of given length and weight."""
    length = request.length
    weight = request.weight
    if weight == 0:
        codewords = [(0,) * length]
    elif weight == length:
        codewords = [(1,) * length]
    else:
        codewords = []
        for ones in combinations(range(length), weight):
            word = [0] * length
            for i in ones:
                word[i] = 1
            codewords.append(tuple(word))
    return ConstantWeightResult(
        codewords=tuple(codewords),
        count=len(codewords),
    )
