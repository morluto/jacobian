"""Typed contracts for 3-term progression hypergraph construction."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.hypergraphs._models import FiniteHypergraph

MAX_GROUP_ORDER: int = 200


class ProgressionHypergraphRequest(StrictModel):
    """Order of the cyclic group Z/nZ for 3-AP hypergraph construction."""

    group_order: int = Field(ge=2, le=MAX_GROUP_ORDER)


class ProgressionHypergraphResult(StrictModel):
    """The 3-uniform hypergraph of all 3-term arithmetic progressions in Z/nZ."""

    group_order: int
    hypergraph: FiniteHypergraph


__all__ = [
    "ProgressionHypergraphRequest",
    "ProgressionHypergraphResult",
    "MAX_GROUP_ORDER",
]
