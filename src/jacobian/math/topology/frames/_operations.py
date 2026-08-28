"""Exact domain functions for finite vector families and frames."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames._arithmetic import dot
from jacobian.math.topology.frames._models import (
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
    return GramResult._from_kernel(vectors=request.vectors, gram=gram)


def _admit_frame(request: FiniteFrameRequest) -> None:
    from sympy import Matrix

    if Matrix(request.vectors).rank() != len(request.vectors[0]):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must span its ambient space",
        )


def compute_coherence(request: CoherenceRequest) -> CoherenceResult:
    _admit_frame(request)
    if any(not any(vector) for vector in request.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="coherence requires every vector to be nonzero",
        )
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
    return CoherenceResult._from_kernel(
        vectors=request.vectors,
        coherence_squared=CanonicalRational.from_fraction(value),
        maximizing_pair=pair,
    )


def compute_frame_potential(request: FiniteFrameRequest) -> FramePotentialResult:
    _admit_frame(request)
    total = sum(
        dot(left, right) ** 2 for left in request.vectors for right in request.vectors
    )
    return FramePotentialResult._from_kernel(
        vectors=request.vectors, potential=format_canonical_integer(total)
    )
