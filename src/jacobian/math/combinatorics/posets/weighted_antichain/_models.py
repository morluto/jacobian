"""Typed contracts for the weighted-antichain operation."""

from jacobian._models import StrictModel
from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.posets.core._models import FinitePoset


class WeightedAntichainRequest(StrictModel):
    """Request the maximum-weight antichain of a finite poset."""

    poset: FinitePoset
    weights: tuple[CanonicalRational, ...]


class WeightedAntichainResult(StrictModel):
    """The exact maximum-weight antichain."""

    poset_digest: str
    weights: tuple[CanonicalRational, ...]
    maximum_weight: CanonicalRational
    maximum_antichain: tuple[str, ...]
    method: str


__all__ = [
    "WeightedAntichainRequest",
    "WeightedAntichainResult",
]
