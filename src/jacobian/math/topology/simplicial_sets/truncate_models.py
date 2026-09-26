"""Contracts for truncating a finite simplicial-set prefix."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICIAL_SET_DEGREE,
    FiniteTruncatedSimplicialSet,
)


class SimplicialSetTruncateRequest(StrictModel):
    simplicial_set: FiniteTruncatedSimplicialSet
    max_degree: StrictInt = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)


__all__ = ["SimplicialSetTruncateRequest"]
