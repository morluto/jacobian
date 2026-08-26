"""Per-request admission calculations for fixed-point prefixes."""

from __future__ import annotations

import json

from jacobian.math.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    ProlongableSubstitution,
    _require_prolongable_source_occurrence_bound,
)

MAX_FIXED_POINT_GENERATION_WORK = 1_000_000
MAX_FIXED_POINT_RESULT_BYTES = 512_000


def require_fixed_point_prefix_budget(
    source: ProlongableSubstitution, prefix_length: int
) -> None:
    """Admit a fixed-point prefix before any iterate is materialized."""

    if not 0 <= prefix_length <= MAX_MORPHISM_OUTPUT_LENGTH:
        raise ValueError(f"prefix length must be in 0..{MAX_MORPHISM_OUTPUT_LENGTH}")
    _require_prolongable_source_occurrence_bound(source)
    generation_work = 4 * prefix_length * prefix_length
    if generation_work > MAX_FIXED_POINT_GENERATION_WORK:
        raise ValueError(
            "fixed-point generation exceeds the work bound "
            f"({generation_work} > {MAX_FIXED_POINT_GENERATION_WORK})"
        )
    result_bytes = _fixed_point_result_byte_bound(source, prefix_length)
    if result_bytes > MAX_FIXED_POINT_RESULT_BYTES:
        raise ValueError(
            "fixed-point result exceeds the byte bound "
            f"({result_bytes} > {MAX_FIXED_POINT_RESULT_BYTES})"
        )


def _fixed_point_result_byte_bound(
    source: ProlongableSubstitution, prefix_length: int
) -> int:
    alphabet = source.substitution.morphism.target_alphabet
    encoded_symbols = tuple(
        len(json.dumps(symbol, ensure_ascii=True).encode("utf-8"))
        for symbol in alphabet
    )
    prefix_bytes = (
        128
        + sum(encoded_symbols)
        + len(encoded_symbols)
        + prefix_length * (max(encoded_symbols) + 1)
    )
    ledger_length = max(1, prefix_length)
    ledger_bytes = 128 + ledger_length * (len(str(max(1, prefix_length))) + 1)
    return (
        4_096
        + len(source.model_dump_json().encode("utf-8"))
        + prefix_bytes
        + ledger_bytes
    )
