"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology.frames._flint import integer_gram, integer_gram_and_rank
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    FramePotentialResult,
    GramResult,
)
from jacobian.math.topology.frames.values import VectorFamily

__all__ = ["coherence", "frame_potential", "gram", "verify_gram"]


MAX_FRAME_GRAM_ENTRIES = 2_097_152
MAX_FRAME_GRAM_MULTIPLY_ADDS = 536_870_912


def _require_gram_work_budget(value: VectorFamily) -> None:
    vector_count = len(value.vectors)
    dimension = value.dimension
    gram_entries = vector_count**2
    if gram_entries > MAX_FRAME_GRAM_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("vectors",),
            code="frames.gram_intermediate_budget",
            message="frame Gram intermediate exceeds its entry budget",
        )
    if gram_entries * dimension > MAX_FRAME_GRAM_MULTIPLY_ADDS:
        raise OperationResourceAdmissionError(
            location=("vectors",),
            code="frames.gram_work_budget",
            message="frame Gram computation exceeds its multiply-add work budget",
        )


def _gram_result(value: VectorFamily) -> GramResult:
    matrix = integer_gram(value.vectors)
    return GramResult._from_kernel(
        vectors=value.vectors,
        dimension=value.dimension,
        gram=IntegerMatrix(
            row_count=len(matrix),
            column_count=len(matrix[0]) if matrix else len(value.vectors),
            entries=matrix,
        ),
    )


def _admit_frame(value: VectorFamily, *, rank: int) -> None:
    if rank != value.dimension:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must span its ambient space",
        )


def _admit_frame_shape(value: VectorFamily) -> None:
    if len(value.vectors) < value.dimension:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must have at least as many vectors as coordinates",
        )


def gram(value: VectorFamily) -> GramResult:
    """Compute the exact Gram matrix of a vector family."""
    _require_gram_work_budget(value)
    return _gram_result(value)


def verify_gram(claim: GramResult) -> bool:
    """Verify the retained vector-family Gram relation for a decoded claim."""
    try:
        if (
            claim.gram.row_count != len(claim.vectors)
            or claim.gram.column_count != len(claim.vectors)
            or any(len(vector) != claim.dimension for vector in claim.vectors)
        ):
            return False
    except (AttributeError, IndexError, TypeError):
        return False
    _require_gram_work_budget(claim)
    return claim.gram.entries == integer_gram(claim.vectors)


def coherence(value: VectorFamily) -> CoherenceResult:
    """Compute exact normalized squared coherence of a finite frame."""
    _require_gram_work_budget(value)
    if any(not any(vector) for vector in value.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="coherence requires every vector to be nonzero",
        )
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors, dimension=value.dimension)
    _admit_frame(value, rank=rank)
    assert matrix is not None
    maximum_numerator = 0
    maximum_denominator = 1
    pair: tuple[int, int] | None = None
    for left in range(len(value.vectors)):
        for right in range(left + 1, len(value.vectors)):
            inner_product = matrix[left][right]
            denominator = matrix[left][left] * matrix[right][right]
            numerator = inner_product * inner_product
            candidate_pair = (left, right)
            comparison = numerator * maximum_denominator - (
                maximum_numerator * denominator
            )
            if comparison > 0 or (
                comparison == 0 and (pair is None or candidate_pair > pair)
            ):
                maximum_numerator = numerator
                maximum_denominator = denominator
                pair = candidate_pair
    return CoherenceResult._from_kernel(
        vectors=value.vectors,
        dimension=value.dimension,
        coherence_squared=CanonicalRational.from_fraction(
            Fraction(maximum_numerator, maximum_denominator)
        ),
        maximizing_pair=pair,
    )


def frame_potential(value: VectorFamily) -> FramePotentialResult:
    """Compute the exact frame potential of a finite frame."""
    _require_gram_work_budget(value)
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors, dimension=value.dimension)
    _admit_frame(value, rank=rank)
    assert matrix is not None
    total = sum(entry**2 for row in matrix for entry in row)
    return FramePotentialResult._from_kernel(
        vectors=value.vectors, dimension=value.dimension, potential=total
    )
