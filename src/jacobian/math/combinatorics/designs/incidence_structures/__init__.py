"""Supported native exact finite-incidence APIs."""

from jacobian.math.combinatorics.designs.incidence_structures._models import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceStructure,
    IncidenceTradeResult,
)
from jacobian.math.combinatorics.designs.incidence_structures.operations import (
    check_incidence_trade,
    complement,
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
    "check_incidence_trade",
    "complement",
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
