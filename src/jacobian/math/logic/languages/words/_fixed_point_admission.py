"""Per-request admission calculations for fixed-point prefixes."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.languages.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    ProlongableSubstitution,
    _require_growing_seed,
    _require_prolongable_source_occurrence_bound,
)

MAX_FIXED_POINT_GENERATION_WORK = 1_000_000


def require_fixed_point_prefix_budget(
    source: ProlongableSubstitution, prefix_length: int
) -> None:
    """Admit a fixed-point prefix before any iterate is materialized."""

    if not 0 <= prefix_length <= MAX_MORPHISM_OUTPUT_LENGTH:
        raise PydanticCustomError(
            "word.prefix_length",
            f"prefix length must be in 0..{MAX_MORPHISM_OUTPUT_LENGTH}",
        )
    _require_prolongable_source_occurrence_bound(source)
    _require_growing_seed(source)
    generation_work = 4 * prefix_length * prefix_length
    if generation_work > MAX_FIXED_POINT_GENERATION_WORK:
        raise OperationResourceAdmissionError(
            location=("prefix_length",),
            code="words.fixed_point_generation_budget",
            message="fixed-point generation exceeds the work bound "
            f"({generation_work} > {MAX_FIXED_POINT_GENERATION_WORK})",
        )
