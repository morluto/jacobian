"""Native operations over canonical topology values."""

from __future__ import annotations

from jacobian.math.topology._models import ChainComplexResult
from jacobian.math.topology._simplicial_kernel import (
    barycentric_subdivision,
    canonicalize,
    chain_complex,
    homology,
    integral_homology,
    pseudomanifold,
    shelling_check,
)
from jacobian.math.topology.chain_complexes.values import ChainComplexValue


def simplicial_chain_complex_value(result: ChainComplexResult) -> ChainComplexValue:
    """Return the canonical chain-complex value carried by a chain result."""
    if result.canonical_value is None:
        raise ValueError(
            "only unreduced prime-field simplicial chain complexes convert "
            "to a canonical chain-complex value"
        )
    return result.canonical_value


__all__ = [
    "barycentric_subdivision",
    "canonicalize",
    "chain_complex",
    "homology",
    "integral_homology",
    "pseudomanifold",
    "shelling_check",
    "simplicial_chain_complex_value",
]
