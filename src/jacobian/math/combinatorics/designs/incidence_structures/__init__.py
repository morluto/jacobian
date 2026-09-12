"""Supported native exact finite-incidence APIs."""

from jacobian.math.combinatorics.designs.incidence_structures._models import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceTradeResult,
    SteinerTripleSystemRequest,
    SteinerTripleSystemResult,
    SteinerTripleSystemShard,
)
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    check_incidence_trade,
    complement,
    construct_steiner_triple_system,
    containment_profile,
    degree_profile,
    derived_residual,
    dual,
    gram,
    incidence_matrix,
    intersections,
    levi_graph,
    restriction,
)

__all__ = [
    "ContainmentProfileResult",
    "IncidenceMomentComparison",
    "IncidenceStructure",
    "IncidenceTradeResult",
    "SteinerTripleSystemRequest",
    "SteinerTripleSystemResult",
    "SteinerTripleSystemShard",
    "check_incidence_trade",
    "complement",
    "construct_steiner_triple_system",
    "containment_profile",
    "degree_profile",
    "derived_residual",
    "dual",
    "gram",
    "incidence_matrix",
    "intersections",
    "levi_graph",
    "restriction",
]
