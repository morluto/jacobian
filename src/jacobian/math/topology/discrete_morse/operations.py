"""Native entry point for discrete Morse matching construction."""

from __future__ import annotations

from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology._request_admission import (
    require_canonical_complex_admission,
)
from jacobian.math.topology.discrete_morse._kernel import (
    construct_matching as _construct_matching,
)
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingResult,
    MatchingPair,
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


__all__ = ["construct_matching"]
