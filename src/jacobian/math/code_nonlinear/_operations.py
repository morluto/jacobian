"""Domain functions for nonlinear binary code operations."""
# mypy: disable-error-code="no-untyped-def,no-untyped-call,return-value"

from __future__ import annotations

from itertools import combinations

from jacobian.math.code_nonlinear._budget import (
    require_constant_weight_admission,
    require_pair_work_admission,
    require_profile_admission,
    require_set_system_output_bound,
)
from jacobian.math.code_nonlinear._models import (
    BinaryCodeRequest,
    ConstantWeightRequest,
    ConstantWeightResult,
    DistanceProfileResult,
)
from jacobian.math.code_nonlinear.values import ExplicitBinaryCode


def compute_distance_profile(request: BinaryCodeRequest) -> DistanceProfileResult:
    """Compute minimum Hamming distance and weight profile of a binary code."""
    code = request.code
    codewords = code.codewords
    require_pair_work_admission(code)
    min_dist = 0
    if len(codewords) > 1:
        min_dist = code.length + 1
    for i, w1 in enumerate(codewords):
        for w2 in codewords[i + 1 :]:
            dist = sum(a != b for a, b in zip(w1, w2, strict=True))
            if dist < min_dist:
                min_dist = dist
    return DistanceProfileResult(
        code=code,
        minimum_distance=min_dist,
        weight_profile=tuple(sum(w) for w in codewords),
        method="EXACT_ENUMERATION",
    )


def compute_constant_weight(request: ConstantWeightRequest) -> ConstantWeightResult:
    """Generate all constant-weight binary words of given length and weight."""
    length = request.length
    weight = request.weight
    expected_count = require_constant_weight_admission(length, weight)
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
    code = ExplicitBinaryCode(length=length, codewords=tuple(codewords))
    if len(code.codewords) != expected_count:
        raise ValueError(
            "constant-weight generation produced an unexpected cardinality"
        )
    return ConstantWeightResult(code=code, count=len(code.codewords))


def _word_distance(
    word1: tuple[int, ...], word2: tuple[int, ...]
) -> tuple[int, tuple[int, ...], int, int, int]:
    """Return (distance, differing_coords, weight1, weight2, support_intersection)."""
    diff = tuple(i for i, (a, b) in enumerate(zip(word1, word2, strict=True)) if a != b)
    w1 = sum(word1)
    w2 = sum(word2)
    inter = sum(1 for a, b in zip(word1, word2, strict=True) if a == 1 and b == 1)
    return len(diff), diff, w1, w2, inter


def _explicit_profile(code: ExplicitBinaryCode):
    """Compute the complete profile of an explicit binary code."""
    codewords = code.codewords
    n = code.length
    m_count = len(codewords)

    weight_distribution = [0] * (n + 1)
    for w in codewords:
        weight_distribution[sum(w)] += 1

    min_dist = 0
    max_dist = 0
    distance_histogram = [0] * (n + 1)
    min_pair = None
    max_pair = None

    for i in range(m_count):
        for j in range(i + 1, m_count):
            dist = sum(a != b for a, b in zip(codewords[i], codewords[j], strict=True))
            distance_histogram[dist] += 1
            if min_dist == 0 or dist < min_dist:
                min_dist = dist
                min_pair = (i, j)
            if dist > max_dist:
                max_dist = dist
                max_pair = (i, j)

    return {
        "weight_distribution": tuple(weight_distribution),
        "minimum_distance": min_dist,
        "maximum_distance": max_dist,
        "distance_histogram": tuple(distance_histogram),
        "min_distance_pair": min_pair,
        "max_distance_pair": max_pair,
    }


def _constant_weight_profile(code: ExplicitBinaryCode):
    """Profile of a constant-weight code using support-intersection distances."""
    codewords = code.codewords
    w = sum(codewords[0]) if codewords else 0
    m_count = len(codewords)

    distance_histogram = [0] * (2 * w + 1)
    min_dist = 0

    for i in range(m_count):
        for j in range(i + 1, m_count):
            inter = sum(
                1
                for a, b in zip(codewords[i], codewords[j], strict=True)
                if a == 1 and b == 1
            )
            dist = 2 * (w - inter)
            distance_histogram[dist] += 1
            if min_dist == 0 or dist < min_dist:
                min_dist = dist

    return {
        "minimum_distance": min_dist,
        "distance_histogram": tuple(distance_histogram),
    }


def _to_set_system(code: ExplicitBinaryCode) -> tuple[tuple[int, ...], ...]:
    """Convert a canonical explicit code to its support subsets."""
    return tuple(
        tuple(index for index, bit in enumerate(word) if bit) for word in code.codewords
    )


def compute_word_distance(request):
    """Compute Hamming distance between two binary words."""
    from jacobian.math.code_nonlinear._models import WordDistanceResult

    dist, diff, w1, w2, inter = _word_distance(request.word1, request.word2)
    return WordDistanceResult(
        word1=request.word1,
        word2=request.word2,
        distance=dist,
        differing_coordinates=diff,
        weight1=w1,
        weight2=w2,
        support_intersection=inter,
    )


def compute_explicit_profile(request):
    """Compute the complete profile of an explicit binary code."""
    from jacobian.math.code_nonlinear._models import ExplicitProfileResult

    require_profile_admission(request.code)
    profile = _explicit_profile(request.code)
    return ExplicitProfileResult(
        code=request.code,
        length=request.code.length,
        cardinality=len(request.code.codewords),
        weight_distribution=profile["weight_distribution"],
        minimum_distance=profile["minimum_distance"],
        maximum_distance=profile["maximum_distance"],
        distance_histogram=profile["distance_histogram"],
        min_distance_pair=profile["min_distance_pair"],
        max_distance_pair=profile["max_distance_pair"],
    )


def compute_constant_weight_profile(request):
    """Profile of a constant-weight binary code."""
    from jacobian.math.code_nonlinear._models import ConstantWeightProfileResult

    require_profile_admission(request.code)
    profile = _constant_weight_profile(request.code)
    return ConstantWeightProfileResult(
        code=request.code,
        length=request.code.length,
        weight=sum(request.code.codewords[0]) if request.code.codewords else 0,
        cardinality=len(request.code.codewords),
        minimum_distance=profile["minimum_distance"],
        distance_histogram=profile["distance_histogram"],
    )


def compute_to_set_system(request):
    """Map codewords to support subsets on coordinate labels."""
    from jacobian.math.code_nonlinear._models import ToSetSystemResult

    require_set_system_output_bound(request.code)
    supports = _to_set_system(request.code)
    return ToSetSystemResult(
        code=request.code,
        length=request.code.length,
        cardinality=len(request.code.codewords),
        supports=supports,
    )


def to_set_system(code: ExplicitBinaryCode) -> tuple[tuple[int, ...], ...]:
    """Convert a canonical code without applying the MCP transport bound."""
    return _to_set_system(code)
