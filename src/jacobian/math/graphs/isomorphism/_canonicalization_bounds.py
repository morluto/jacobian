"""Shared mathematical admission for colored-graph canonicalization."""

from pydantic_core import PydanticCustomError

from jacobian.math.graphs.isomorphism._canonicalization import (
    MAX_CANONICAL_PERMUTATIONS,
    MAX_CANONICALIZATION_WORK,
    canonical_permutation_count,
    canonicalization_work,
)
from jacobian.math.graphs.values import (
    MAX_SIMPLE_GRAPH_VERTICES,
    ColoredUndirectedGraph,
)


def require_admitted_colored_graph_canonicalization(
    graph: ColoredUndirectedGraph,
) -> None:
    """Check the full execution and canonical-result envelope for ``graph``."""

    if len(graph.graph.vertices) > MAX_SIMPLE_GRAPH_VERTICES:
        raise PydanticCustomError(
            "graph.colored_canonicalization.vertex_bound",
            "colored-graph canonicalization supports at most "
            f"{MAX_SIMPLE_GRAPH_VERTICES} vertices",
        )
    candidate_count = canonical_permutation_count(graph)
    if candidate_count > MAX_CANONICAL_PERMUTATIONS:
        raise PydanticCustomError(
            "graph.colored_canonicalization_exceeds_max_canonical_permutations_permutatio",
            "colored-graph canonicalization exceeds the "
            f"{MAX_CANONICAL_PERMUTATIONS}-permutation bound",
        )
    execution_work = canonicalization_work(graph)
    if execution_work > MAX_CANONICALIZATION_WORK:
        raise PydanticCustomError(
            "graph.colored_canonicalization_exceeds_max_canonicalization_work",
            "colored-graph canonicalization exceeds the "
            f"{MAX_CANONICALIZATION_WORK}-unit execution work bound",
        )


__all__ = [
    "require_admitted_colored_graph_canonicalization",
]
