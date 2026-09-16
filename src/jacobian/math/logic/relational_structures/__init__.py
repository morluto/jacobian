"""Exact finite relational structures and homomorphism checking."""

from jacobian.math.logic.relational_structures._models import (
    HomomorphismCheckResult,
    HomomorphismStatus,
    HomomorphismViolationWitness,
    SymbolTransportProfile,
)
from jacobian.math.logic.relational_structures.operations import check_homomorphism
from jacobian.math.logic.relational_structures.values import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)

# The authoritative native surface: every export accepts domain values
# directly. Wire-envelope request handlers live in ``_tools.py`` and are not
# part of this native API.
__all__ = [
    "FiniteRelationSymbol",
    "FiniteRelationalStructure",
    "HomomorphismCheckResult",
    "HomomorphismStatus",
    "HomomorphismViolationWitness",
    "SymbolTransportProfile",
    "check_homomorphism",
]
