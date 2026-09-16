"""Exact discrete Morse theory on finite simplicial complexes."""

from jacobian.math.topology.discrete_morse._models import (
    CriticalCellBasis,
    CriticalCellProfile,
    DiscreteMorseMatchingResult,
    GradientPath,
    GradientPathCount,
    GradientPathsResult,
    GradientPathStep,
    MatchingPair,
    MorseBoundaryEntry,
    MorseComplexResult,
    MorseGradientStepKind,
    MorseMatchingFault,
    MorseMatchingOutcome,
)
from jacobian.math.topology.discrete_morse.operations import (
    compute_gradient_paths,
    compute_morse_complex,
    construct_matching,
)

__all__ = [
    "CriticalCellBasis",
    "CriticalCellProfile",
    "DiscreteMorseMatchingResult",
    "GradientPath",
    "GradientPathCount",
    "GradientPathStep",
    "GradientPathsResult",
    "MatchingPair",
    "MorseBoundaryEntry",
    "MorseComplexResult",
    "MorseGradientStepKind",
    "MorseMatchingFault",
    "MorseMatchingOutcome",
    "compute_gradient_paths",
    "compute_morse_complex",
    "construct_matching",
]
