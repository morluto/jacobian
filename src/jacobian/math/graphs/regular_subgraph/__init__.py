"""k-regular subgraph operations."""

from jacobian.math.graphs.regular_subgraph._models import (
    RegularSubgraphResult,
)
from jacobian.math.graphs.regular_subgraph.operations import (
    find_k_regular_subgraph,
)

__all__ = [
    "RegularSubgraphResult",
    "find_k_regular_subgraph",
]
