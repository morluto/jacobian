"""Exact discrete Morse theory on finite simplicial complexes."""

from jacobian.math.topology.discrete_morse._models import (
    CriticalCellBasis,
    CriticalCellProfile,
    DiscreteMorseMatchingResult,
    GradientPath,
    GradientPathCount,
    GradientPathsResult,
    GradientPathStep,
    IntegerMorseComplexResult,
    MatchingPair,
    MorseBoundaryEntry,
    MorseComplexResult,
    MorseGradientStepKind,
    MorseMatchingFault,
    MorseMatchingOutcome,
)
from jacobian.math.topology.discrete_morse.extensions import (
    CollapseSequenceRequest,
    GreedyCollapseRequest,
    GreedyMatchingRequest,
    collapse_sequence,
    greedy_collapse,
    greedy_matching,
)
from jacobian.math.topology.discrete_morse.operations import (
    compute_gradient_paths,
    compute_integer_morse_complex,
    compute_morse_complex,
    construct_matching,
)

__all__ = [
    "CollapseSequenceRequest",
    "CriticalCellBasis",
    "CriticalCellProfile",
    "DiscreteMorseMatchingResult",
    "GradientPath",
    "GradientPathCount",
    "GradientPathStep",
    "GradientPathsResult",
    "GreedyCollapseRequest",
    "GreedyMatchingRequest",
    "IntegerMorseComplexResult",
    "MatchingPair",
    "MorseBoundaryEntry",
    "MorseComplexResult",
    "MorseGradientStepKind",
    "MorseMatchingFault",
    "MorseMatchingOutcome",
    "collapse_sequence",
    "compute_gradient_paths",
    "compute_integer_morse_complex",
    "compute_morse_complex",
    "construct_matching",
    "greedy_collapse",
    "greedy_matching",
]
