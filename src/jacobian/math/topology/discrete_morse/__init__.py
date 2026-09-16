"""Exact discrete Morse theory on finite simplicial complexes."""

from jacobian.math.topology.discrete_morse._models import (
    CriticalCellProfile,
    DiscreteMorseMatchingRequest,
    DiscreteMorseMatchingResult,
    MatchingPair,
    MorseMatchingFault,
    MorseMatchingOutcome,
)
from jacobian.math.topology.discrete_morse.operations import construct_matching

__all__ = [
    "CriticalCellProfile",
    "DiscreteMorseMatchingRequest",
    "DiscreteMorseMatchingResult",
    "MatchingPair",
    "MorseMatchingFault",
    "MorseMatchingOutcome",
    "construct_matching",
]
