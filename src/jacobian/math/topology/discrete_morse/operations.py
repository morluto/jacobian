"""Native entry points for discrete Morse matching, paths, and Morse complex."""

from __future__ import annotations

from jacobian.math.topology._models import FiniteSimplicialComplex, Simplex
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.discrete_morse._contraction import (
    compute_chain_contraction as _compute_chain_contraction,
)
from jacobian.math.topology.discrete_morse._kernel import (
    compute_gradient_paths as _compute_gradient_paths,
)
from jacobian.math.topology.discrete_morse._kernel import (
    compute_integer_morse_complex as _compute_integer_morse_complex,
)
from jacobian.math.topology.discrete_morse._kernel import (
    compute_morse_complex as _compute_morse_complex,
)
from jacobian.math.topology.discrete_morse._kernel import (
    construct_matching as _construct_matching,
)
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingResult,
    GradientPathsResult,
    IntegerMorseComplexResult,
    MatchingPair,
    MorseChainContractionResult,
    MorseComplexResult,
)


def construct_matching(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> DiscreteMorseMatchingResult:
    """Construct one bounded discrete Morse matching classification.

    The caller supplies the canonical complex; the kernel then decides the
    supplied pairs and never re-runs the face-closure mathematics in a
    validator or transport layer.
    """

    require_canonical_complex_admission(complex_)
    return _construct_matching(complex_, pairs)


def compute_gradient_paths(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
    start: Simplex,
    target: Simplex | None = None,
) -> GradientPathsResult:
    """Enumerate the complete bounded gradient-path family from a critical cell.

    The kernel first re-establishes that the supplied pairs are an acyclic
    matching, then replays every alternating down/up V-path from the selected
    critical cell to its critical targets of the adjacent lower dimension.
    """

    require_canonical_complex_admission(complex_)
    return _compute_gradient_paths(complex_, pairs, start, target)


def compute_morse_complex(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> MorseComplexResult:
    """Compute the graded GF(2) Morse complex of a bounded acyclic matching.

    The kernel re-establishes acyclicity, grades the critical cells by
    dimension, and derives every reduced boundary coefficient from complete
    gradient-path counts.
    """

    require_canonical_complex_admission(complex_)
    return _compute_morse_complex(complex_, pairs)


def compute_integer_morse_complex(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> IntegerMorseComplexResult:
    """Compute the ZZ Morse chain complex using lex-oriented signed paths."""

    require_canonical_complex_admission(complex_)
    return _compute_integer_morse_complex(complex_, pairs)


def compute_chain_contraction(
    complex_: FiniteSimplicialComplex,
    pairs: tuple[MatchingPair, ...],
) -> MorseChainContractionResult:
    """Return a bounded integral chain contraction onto the Morse complex."""

    return _compute_chain_contraction(complex_, pairs)


__all__ = [
    "compute_chain_contraction",
    "compute_gradient_paths",
    "compute_integer_morse_complex",
    "compute_morse_complex",
    "construct_matching",
]
