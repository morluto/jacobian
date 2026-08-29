"""Typed contracts for the antichain enumeration operation."""

from jacobian._models import StrictModel
from jacobian.math.combinatorics.posets.core._models import FinitePoset

MAX_ELEMENTS = 24
MAX_ANTICHAINS = 50000


class AntichainEnumerationRequest(StrictModel):
    """Request to enumerate antichains of specified cardinalities."""

    poset: FinitePoset
    min_cardinality: int = 1
    max_cardinality: int = 1


class AntichainEnumerationResult(StrictModel):
    """A complete enumeration of antichains in the requested cardinality range."""

    poset_digest: str
    min_cardinality: int
    max_cardinality: int
    antichains: tuple[tuple[str, ...], ...]
    count: int


__all__ = [
    "MAX_ANTICHAINS",
    "MAX_ELEMENTS",
    "AntichainEnumerationRequest",
    "AntichainEnumerationResult",
]
