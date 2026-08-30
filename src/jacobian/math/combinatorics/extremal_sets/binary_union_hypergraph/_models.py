"""Typed contracts for the binary-union relation hypergraph."""

import math
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_BINARY_UNION_WORK = 2_000_000


def _binary_union_admission_error(
    sets: tuple[tuple[int, ...], ...],
) -> tuple[str, str] | None:
    if len(sets) > MAX_VERTICES:
        return ("carrier_bound", f"family exceeds the {MAX_VERTICES}-vertex bound")
    if any(values != tuple(sorted(set(values))) for values in sets):
        return ("canonical_sets", "each finite set must be strictly increasing")
    if len(set(sets)) != len(sets):
        return ("distinct_sets", "the finite-set family must contain distinct sets")
    candidate_pairs = math.comb(len(sets), 2)
    if candidate_pairs > MAX_EDGES or 3 * candidate_pairs > MAX_TOTAL_INCIDENCES:
        return ("output_bound", "binary-union relation family exceeds result bounds")
    work = max(0, len(sets) - 1) * sum(len(values) for values in sets)
    if work > MAX_BINARY_UNION_WORK:
        return ("work_bound", "binary-union relation scan exceeds the work bound")
    return None


class BinaryUnionHypergraphRequest(StrictModel):
    """Request to construct the binary-union relation hypergraph."""

    sets: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_bounded_canonical_family(self) -> Self:
        failure = _binary_union_admission_error(self.sets)
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"binary_union.{code}", message)
        return self


class BinaryUnionHypergraphResult(StrictModel):
    """The 3-uniform binary-union relation hypergraph."""

    sets: tuple[tuple[int, ...], ...]
    hypergraph: FiniteHypergraph
    relation_count: int


__all__ = [
    "MAX_BINARY_UNION_WORK",
    "BinaryUnionHypergraphRequest",
    "BinaryUnionHypergraphResult",
    "_binary_union_admission_error",
]
