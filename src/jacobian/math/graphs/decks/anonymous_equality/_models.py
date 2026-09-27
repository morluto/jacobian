"""Contracts for equality of anonymous graph-card multisets."""

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultiset


class AnonymousDeckEqualityRequest(StrictModel):
    """Compare two finite anonymous multisets of same-order graph cards."""

    left: AnonymousGraphCardMultiset
    right: AnonymousGraphCardMultiset


class AnonymousDeckEqualityResult(StrictModel):
    """Whether the two card multisets have equal isomorphism-class counts."""

    equal: bool = Field(description="Exact multiset equality up to card isomorphism.")


__all__ = ["AnonymousDeckEqualityRequest", "AnonymousDeckEqualityResult"]
