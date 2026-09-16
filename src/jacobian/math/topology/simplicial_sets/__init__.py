"""Finite truncated simplicial sets with exact face/degeneracy tables."""

from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
    SimplicialIdentityObstruction,
    SimplicialSetTablesRequest,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables

__all__ = [
    "FiniteTruncatedSimplicialSet",
    "SimplicialIdentityObstruction",
    "SimplicialSetTablesRequest",
    "SimplicialSetTablesResult",
    "from_tables",
]
