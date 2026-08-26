"""Neutral admission bounds for coherent-configuration values.

These quantities depend only on the canonical complete pair partition.  They
are shared by value parsing and the analysis kernel, so neither side needs to
re-enter the public operation layer.
"""

from __future__ import annotations

import json

from jacobian.math.coherent_configurations.values import (
    MAX_ANALYSIS_WORK,
    MAX_COHERENT_CONFIGURATION_RESULT_BYTES,
    MAX_COHERENT_CONFIGURATION_SOURCE_BYTES,
    CoherentConfigurationInput,
)


def _source_bytes(source: CoherentConfigurationInput) -> int:
    return len(source.model_dump_json().encode("utf-8"))


def _json_string_byte_bound(value: str) -> int:
    """Conservatively bound one JSON string scalar, including escapes."""

    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def estimate_analysis_result_bytes(source: CoherentConfigurationInput) -> int:
    """Bound the full wire result before materializing its cubic tensor."""

    source_bytes = _source_bytes(source)
    relation_id_bytes = max(
        _json_string_byte_bound(value) for value in source.relation_ids
    )
    point_bytes = max(_json_string_byte_bound(value) for value in source.points)
    relation_count = len(source.relation_ids)
    point_count = len(source.points)
    tensor_entry_bytes = 3 * relation_id_bytes + 128
    fibre_bytes = point_count * (point_bytes + 48)
    transpose_bytes = relation_count * (2 * relation_id_bytes + 64)
    return (
        2 * source_bytes
        + relation_count**3 * tensor_entry_bytes
        + fibre_bytes
        + transpose_bytes
        + 2_048
    )


def require_analysis_admission(source: CoherentConfigurationInput) -> None:
    """Reject source or predicted analysis/result work before cubic expansion."""

    if _source_bytes(source) > MAX_COHERENT_CONFIGURATION_SOURCE_BYTES:
        raise ValueError("coherent-configuration source exceeds the byte budget")
    point_count = len(source.points)
    relation_count = len(source.relation_ids)
    work = 4 * relation_count**2 * point_count**3
    if work > MAX_ANALYSIS_WORK:
        raise ValueError("coherent-configuration analysis exceeds the work budget")
    if estimate_analysis_result_bytes(source) > MAX_COHERENT_CONFIGURATION_RESULT_BYTES:
        raise ValueError("coherent-configuration result exceeds the byte budget")


__all__ = ["estimate_analysis_result_bytes", "require_analysis_admission"]
