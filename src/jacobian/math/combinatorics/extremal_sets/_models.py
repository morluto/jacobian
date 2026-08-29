"""Typed contracts for binary-union relation hypergraphs."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.combinatorics.codes.nonlinear._models import ToSetSystemResult
from jacobian.math.combinatorics.extremal_sets.values import (
    MAX_FAMILY_SIZE,
    IndexedFiniteSetFamily,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    FiniteHypergraph,
)


class BinaryUnionRelationRequest(StrictModel):
    """Request to compute the binary-union relation of an indexed family."""

    source: IndexedFiniteSetFamily | ToSetSystemResult


class UnionRelationRow(StrictModel):
    """One oriented equation ``source[i] union source[j] = source[k]``."""

    edge_id: str
    operand_i: int = Field(ge=0, le=MAX_FAMILY_SIZE - 1)
    operand_j: int = Field(ge=0, le=MAX_FAMILY_SIZE - 1)
    result_k: int = Field(ge=0, le=MAX_FAMILY_SIZE - 1)


class BinaryUnionRelationResult(StrictModel):
    """The complete source-bound binary-union relation."""

    source: IndexedFiniteSetFamily
    rows: tuple[UnionRelationRow, ...] = Field(max_length=MAX_EDGES)
    hypergraph: FiniteHypergraph


__all__ = [
    "BinaryUnionRelationRequest",
    "BinaryUnionRelationResult",
    "UnionRelationRow",
]
