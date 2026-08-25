"""Exact domain functions for finite vector families and frames."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.frames._arithmetic import dot
from jacobian.math.frames._models import (
    CoherenceRequest,
    CoherenceResult,
    FiniteFrameRequest,
    FramePotentialResult,
    GramResult,
    VectorFamilyRequest,
)


def compute_gram(request: VectorFamilyRequest) -> GramResult:
    gram = tuple(
        tuple(dot(left, right) for right in request.vectors) for left in request.vectors
    )
    return GramResult(
        **request.model_dump(), gram=gram, dimension=len(request.vectors[0])
    )


def compute_coherence(request: CoherenceRequest) -> CoherenceResult:
    candidates = []
    for left in range(len(request.vectors)):
        for right in range(left + 1, len(request.vectors)):
            inner_product = dot(request.vectors[left], request.vectors[right])
            denominator = dot(request.vectors[left], request.vectors[left]) * dot(
                request.vectors[right], request.vectors[right]
            )
            candidates.append(
                (Fraction(inner_product * inner_product, denominator), (left, right))
            )
    value, pair = max(candidates, default=(Fraction(0), None))
    return CoherenceResult(
        **request.model_dump(),
        coherence_squared=CanonicalRational.from_fraction(value),
        maximizing_pair=pair,
    )


def compute_frame_potential(request: FiniteFrameRequest) -> FramePotentialResult:
    total = sum(
        dot(left, right) ** 2 for left in request.vectors for right in request.vectors
    )
    return FramePotentialResult(
        **request.model_dump(), potential=format_canonical_integer(total)
    )
