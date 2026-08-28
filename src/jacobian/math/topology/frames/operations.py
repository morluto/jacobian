"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames._arithmetic import dot
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    FramePotentialResult,
    GramResult,
)
from jacobian.math.topology.frames.values import VectorFamily

__all__ = ["coherence", "frame_potential", "gram"]


def _admit_frame(value: VectorFamily) -> None:
    from sympy import Matrix

    if Matrix(value.vectors).rank() != len(value.vectors[0]):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must span its ambient space",
        )


def gram(value: VectorFamily) -> GramResult:
    """Compute the exact Gram matrix of a vector family."""
    matrix = tuple(
        tuple(dot(left, right) for right in value.vectors) for left in value.vectors
    )
    return GramResult._from_kernel(vectors=value.vectors, gram=matrix)


def coherence(value: VectorFamily) -> CoherenceResult:
    """Compute exact normalized squared coherence of a finite frame."""
    _admit_frame(value)
    if any(not any(vector) for vector in value.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="coherence requires every vector to be nonzero",
        )
    candidates = []
    for left in range(len(value.vectors)):
        for right in range(left + 1, len(value.vectors)):
            inner_product = dot(value.vectors[left], value.vectors[right])
            denominator = dot(value.vectors[left], value.vectors[left]) * dot(
                value.vectors[right], value.vectors[right]
            )
            candidates.append(
                (Fraction(inner_product * inner_product, denominator), (left, right))
            )
    maximum, pair = max(candidates, default=(Fraction(0), None))
    return CoherenceResult._from_kernel(
        vectors=value.vectors,
        coherence_squared=CanonicalRational.from_fraction(maximum),
        maximizing_pair=pair,
    )


def frame_potential(value: VectorFamily) -> FramePotentialResult:
    """Compute the exact frame potential of a finite frame."""
    _admit_frame(value)
    total = sum(
        dot(left, right) ** 2 for left in value.vectors for right in value.vectors
    )
    return FramePotentialResult._from_kernel(
        vectors=value.vectors, potential=format_canonical_integer(total)
    )
