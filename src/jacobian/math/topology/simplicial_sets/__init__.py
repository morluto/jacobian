"""Finite truncated simplicial sets with exact face/degeneracy tables."""

from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
    SimplicialIdentityObstruction,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

__all__ = [
    "FiniteTruncatedSimplicialSet",
    "SimplicialIdentityObstruction",
    "SimplicialSetTablesResult",
    "from_tables",
]
