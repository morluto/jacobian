"""Supported native exact finite-incidence APIs."""

from jacobian.math.incidence_structures._models import (
    ContainmentProfileResult,
    IncidenceMomentComparison,
    IncidenceMultiplicityDifference,
    IncidenceStructure,
    IncidenceTradeResult,
)
from jacobian.math.incidence_structures.operations import (
    check_incidence_trade,
    containment_profile,
)

__all__ = [
    "ContainmentProfileResult",
    "IncidenceMomentComparison",
    "IncidenceMultiplicityDifference",
    "IncidenceStructure",
    "IncidenceTradeResult",
    "check_incidence_trade",
    "containment_profile",
]
