"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames._flint import integer_gram, integer_rank
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    FramePotentialResult,
    GramResult,
)
from jacobian.math.topology.frames.values import VectorFamily

__all__ = ["coherence", "frame_potential", "gram"]


_RESULT_RESERVE_BYTES = 128


def _retained_source_bytes(value: VectorFamily) -> int:
    return len(encode_strict_json(value.model_dump(mode="json")))


def _coefficient_height(value: VectorFamily) -> int:
    return max(abs(entry) for vector in value.vectors for entry in vector)


def _gram_result_bound(value: VectorFamily) -> int:
    vectors = value.vectors
    vector_count = len(vectors)
    occupancy = [0] * len(vectors[0])
    max_squared_norm = 0
    for vector in vectors:
        squared_norm = 0
        for column, entry in enumerate(vector):
            squared_norm += entry * entry
            if entry:
                occupancy[column] += 1
        if squared_norm > max_squared_norm:
            max_squared_norm = squared_norm
    cell_count = vector_count * vector_count
    nonzero_cells = min(cell_count, sum(count * count for count in occupancy))
    zero_cells = cell_count - nonzero_cells
    gram_value_chars = len(str(max_squared_norm)) + int(max_squared_norm > 0)
    gram_bytes = (
        nonzero_cells * (gram_value_chars + 1) + zero_cells * 2 + 2 * vector_count
    )
    return _retained_source_bytes(value) + gram_bytes + _RESULT_RESERVE_BYTES


def _compact_result_bound(value: VectorFamily) -> int:
    vector_count = len(value.vectors)
    dimension = len(value.vectors[0])
    gram_bound = dimension * _coefficient_height(value) ** 2
    scalar_chars = len(str(vector_count**2 * gram_bound**2))
    return _retained_source_bytes(value) + scalar_chars * 2 + _RESULT_RESERVE_BYTES


def _require_result_budget(predicted_bytes: int) -> None:
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.result_byte_budget",
            message="frame operation exceeds the canonical result-byte budget",
        )


def _admit_frame(value: VectorFamily, *, rank: int) -> None:
    if rank != len(value.vectors[0]):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must span its ambient space",
        )


def gram(value: VectorFamily) -> GramResult:
    """Compute the exact Gram matrix of a vector family."""
    _require_result_budget(_gram_result_bound(value))
    matrix = integer_gram(value.vectors)
    return GramResult._from_kernel(vectors=value.vectors, gram=matrix)


def coherence(value: VectorFamily) -> CoherenceResult:
    """Compute exact normalized squared coherence of a finite frame."""
    _require_result_budget(_compact_result_bound(value))
    if any(not any(vector) for vector in value.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="coherence requires every vector to be nonzero",
        )
    _admit_frame(value, rank=integer_rank(value.vectors))
    matrix = integer_gram(value.vectors)
    maximum = Fraction(0)
    pair: tuple[int, int] | None = None
    for left in range(len(value.vectors)):
        for right in range(left + 1, len(value.vectors)):
            inner_product = matrix[left][right]
            denominator = matrix[left][left] * matrix[right][right]
            candidate = Fraction(inner_product * inner_product, denominator)
            candidate_pair = (left, right)
            if candidate > maximum or (
                candidate == maximum and (pair is None or candidate_pair > pair)
            ):
                maximum = candidate
                pair = candidate_pair
    return CoherenceResult._from_kernel(
        vectors=value.vectors,
        coherence_squared=CanonicalRational.from_fraction(maximum),
        maximizing_pair=pair,
    )


def frame_potential(value: VectorFamily) -> FramePotentialResult:
    """Compute the exact frame potential of a finite frame."""
    _require_result_budget(_compact_result_bound(value))
    _admit_frame(value, rank=integer_rank(value.vectors))
    matrix = integer_gram(value.vectors)
    total = sum(entry**2 for row in matrix for entry in row)
    return FramePotentialResult._from_kernel(
        vectors=value.vectors, potential=format_canonical_integer(total)
    )
