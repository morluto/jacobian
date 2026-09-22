"""Exact finite relational structures with homomorphism check and search."""

from jacobian.math.logic.relational_structures._models import (
    EmbeddingSearchResult,
    HomomorphismCheckResult,
    HomomorphismCoreResult,
    HomomorphismCountResult,
    HomomorphismSearchResult,
    HomomorphismSearchStatus,
    HomomorphismStatus,
    HomomorphismViolationWitness,
    InducedEmbeddingCheckResult,
    InducedRelationProfile,
    SymbolTransportProfile,
)
from jacobian.math.logic.relational_structures.operations import (
    check_homomorphism,
    compute_core,
    count_homomorphisms,
    search_embedding,
    search_homomorphism,
)
from jacobian.math.logic.relational_structures.values import (
    FiniteRelationalStructure,
    FiniteRelationSymbol,
)

# The authoritative native surface: every export accepts domain values
# directly. Wire-envelope request handlers live in ``_tools.py`` and are not
# part of this native API.
__all__ = [
    "EmbeddingSearchResult",
    "FiniteRelationSymbol",
    "FiniteRelationalStructure",
    "HomomorphismCheckResult",
    "HomomorphismCoreResult",
    "HomomorphismCountResult",
    "HomomorphismSearchResult",
    "HomomorphismSearchStatus",
    "HomomorphismStatus",
    "HomomorphismViolationWitness",
    "InducedEmbeddingCheckResult",
    "InducedRelationProfile",
    "SymbolTransportProfile",
    "check_homomorphism",
    "compute_core",
    "count_homomorphisms",
    "search_embedding",
    "search_homomorphism",
]
