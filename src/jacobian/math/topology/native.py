"""Native topology functions exposing cross-domain canonical values."""

from __future__ import annotations

from jacobian.math.topology._models import ChainComplexResult
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

__all__ = ["simplicial_chain_complex_value"]


def simplicial_chain_complex_value(result: ChainComplexResult) -> ChainComplexValue:
    """Return the canonical value carried by a simplicial chain result."""
    from jacobian.math.topology._operations import (
        _canonical_chain_complex_value,
    )

    return _canonical_chain_complex_value(result)
