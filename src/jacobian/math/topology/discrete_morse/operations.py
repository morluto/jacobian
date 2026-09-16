"""Native entry point for discrete Morse matching construction."""

from __future__ import annotations

from jacobian.math.topology._models import (
    SimplicialComplexCanonicalizationResult,
)
from jacobian.math.topology.discrete_morse._kernel import (
    construct_matching as _construct_matching,
)
from jacobian.math.topology.discrete_morse._models import (
    DiscreteMorseMatchingRequest,
    DiscreteMorseMatchingResult,
)
from jacobian.math.topology.operations import canonicalize


def construct_matching(
    request: DiscreteMorseMatchingRequest,
) -> DiscreteMorseMatchingResult:
    """Construct one bounded discrete Morse matching classification.

    The canonical complex is established once during admission; the kernel
    then decides the supplied pairs and never re-runs that mathematics in a
    validator or transport layer.
    """

    canonical: SimplicialComplexCanonicalizationResult = canonicalize(
        request.complex.vertices, request.complex.facets
    )
    return _construct_matching(canonical.complex, request)


__all__ = ["construct_matching"]
