"""Native operations on canonical finite vector-family values."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.frames._flint import integer_gram, integer_gram_and_rank
from jacobian.math.topology.frames._models import (
    CoherenceResult,
    FramePotentialResult,
    GramResult,
)
from jacobian.math.topology.frames.values import VectorFamily

__all__ = ["coherence", "frame_potential", "gram"]


_RESULT_RESERVE_BYTES = 128
MAX_FRAME_GRAM_ENTRIES = 2_097_152
MAX_FRAME_GRAM_MULTIPLY_ADDS = 536_870_912
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


def _retained_source_bytes(value: VectorFamily) -> int:
    return len(encode_strict_json(value.model_dump(mode="json")))


def _coefficient_height(value: VectorFamily) -> int:
    return max(abs(entry) for vector in value.vectors for entry in vector)


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


def _gram_minimum_result_bound(value: VectorFamily) -> int:
    """Bound the unavoidable source and matrix structure before the kernel."""

    vector_count = len(value.vectors)
    return _retained_source_bytes(value) + 2 * vector_count**2 + 2 * vector_count


def _require_gram_work_budget(value: VectorFamily) -> None:
    vector_count = len(value.vectors)
    dimension = len(value.vectors[0])
    gram_entries = vector_count**2
    if gram_entries > MAX_FRAME_GRAM_ENTRIES:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.gram_intermediate_budget",
            message="frame Gram intermediate exceeds its entry budget",
        )
    if gram_entries * dimension > MAX_FRAME_GRAM_MULTIPLY_ADDS:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.gram_work_budget",
            message="frame Gram computation exceeds its multiply-add work budget",
        )


def _require_gram_entry_representation(value: VectorFamily) -> None:
    gram_entry_bound = max(
        sum(entry**2 for entry in vector) for vector in value.vectors
    )
    if gram_entry_bound > _MAX_SAFE_JSON_INTEGER:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.gram_entry_representation",
            message="frame Gram entries exceed the canonical integer representation",
        )


def _gram_result(value: VectorFamily) -> GramResult:
    matrix = integer_gram(value.vectors)
    return GramResult._from_kernel(vectors=value.vectors, gram=matrix)


def _gram_result_bytes(result: GramResult) -> int:
    output_limit = CanonicalLimits().max_output_bytes
    measurement_limits = CanonicalLimits(max_output_bytes=2 * output_limit)
    try:
        return len(
            encode_strict_json(
                result.model_dump(mode="json"),
                limits=measurement_limits,
            )
        )
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.result_byte_budget",
            message="frame operation exceeds the canonical result-byte budget",
        ) from exc


def _admit_frame(value: VectorFamily, *, rank: int) -> None:
    if rank != len(value.vectors[0]):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must span its ambient space",
        )


def _admit_frame_shape(value: VectorFamily) -> None:
    if len(value.vectors) < len(value.vectors[0]):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.frame_does_not_span",
            message="a finite frame must have at least as many vectors as coordinates",
        )


def gram(value: VectorFamily) -> GramResult:
    """Compute the exact Gram matrix of a vector family."""
    _require_gram_work_budget(value)
    return _gram_result(value)


def coherence(value: VectorFamily) -> CoherenceResult:
    """Compute exact normalized squared coherence of a finite frame."""
    _require_gram_work_budget(value)
    _require_result_budget(_compact_result_bound(value))
    if any(not any(vector) for vector in value.vectors):
        raise OperationDomainValidationError(
            location=("vectors",),
            code="frames.zero_vector",
            message="coherence requires every vector to be nonzero",
        )
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors)
    _admit_frame(value, rank=rank)
    assert matrix is not None
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
    _require_gram_work_budget(value)
    _require_result_budget(_compact_result_bound(value))
    _admit_frame_shape(value)
    rank, matrix = integer_gram_and_rank(value.vectors)
    _admit_frame(value, rank=rank)
    assert matrix is not None
    total = sum(entry**2 for row in matrix for entry in row)
    return FramePotentialResult._from_kernel(
        vectors=value.vectors, potential=format_canonical_integer(total)
    )
