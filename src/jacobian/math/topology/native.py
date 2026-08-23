"""Native topology functions exposing cross-domain canonical values."""

from __future__ import annotations

from jacobian.math.chain_complexes.values import ChainComplexValue
from jacobian.math.topology._models import ChainComplexResult

__all__ = ["simplicial_chain_complex_value"]


def simplicial_chain_complex_value(result: ChainComplexResult) -> ChainComplexValue:
    """The canonical chain-complex value of one simplicial chain complex.

    Accepts the producer's ``ChainComplexResult`` unchanged and returns
    the domain-owned ``ChainComplexValue`` consumed by homology, tensor,
    map, and cone operations. Producer results already carry this value
    as ``canonical_value``; this wrapper serves callers holding an older
    deserialized result.
    """
    from jacobian.math.topology._operations import (
        _canonical_chain_complex_value,
    )

    if result.canonical_value is not None:
        return result.canonical_value
    return _canonical_chain_complex_value(result)
