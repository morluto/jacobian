"""Per-request admission calculations for fixed-point prefixes."""

from __future__ import annotations

from jacobian.math.logic.languages.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    ProlongableSubstitution,
    _require_prolongable_source_occurrence_bound,
)

MAX_FIXED_POINT_GENERATION_WORK = 1_000_000


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
