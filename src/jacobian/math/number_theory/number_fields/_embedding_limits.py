"""Shared resource bounds for exact number-field embedding profiles."""

from __future__ import annotations

from jacobian.canonical import CanonicalLimits

MAX_NUMBER_FIELD_EMBEDDING_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS = 2_097_152
MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS = 32_768


__all__ = [
    "MAX_NUMBER_FIELD_EMBEDDING_RESULT_BYTES",
    "MAX_NUMBER_FIELD_REAL_PART_RESULTANT_STORAGE_BITS",
    "MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS",
]
