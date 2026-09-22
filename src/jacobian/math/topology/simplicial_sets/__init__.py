"""Finite truncated simplicial sets with exact face/degeneracy tables."""

from jacobian.math.topology.simplicial_sets._models import (
    FiniteTruncatedSimplicialSet,
    SimplicialIdentityObstruction,
    SimplicialSetTablesResult,
)
from jacobian.math.topology.simplicial_sets.maps import (
    normalized_chains,
    simplicial_map,
)
from jacobian.math.topology.simplicial_sets.operations import from_tables
from jacobian.math.topology.simplicial_sets.standard import (
    simplex_boundary,
    simplex_horn,
    standard_simplex,
)

__all__ = [
    "FiniteTruncatedSimplicialSet",
    "SimplicialIdentityObstruction",
    "SimplicialSetTablesResult",
    "from_tables",
    "normalized_chains",
    "simplex_boundary",
    "simplex_horn",
    "simplicial_map",
    "standard_simplex",
]
