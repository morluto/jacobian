"""Shared mathematical admission for colored-graph canonicalization."""

from pydantic_core import PydanticCustomError

from jacobian.math.graphs.isomorphism._canonicalization import (
    MAX_CANONICAL_PERMUTATIONS,
    MAX_CANONICAL_REPLAY_WORK,
    MAX_CANONICALIZATION_RESULT_BYTES,
    canonical_permutation_count,
    canonical_replay_work,
    canonicalization_result_wire_bytes,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


def require_admitted_colored_graph_canonicalization(
    graph: ColoredUndirectedGraph,
) -> None:
    """Check the full execution and canonical-result envelope for ``graph``."""

    candidate_count = canonical_permutation_count(graph)
    if candidate_count > MAX_CANONICAL_PERMUTATIONS:
        raise PydanticCustomError(
            "graph.colored_canonicalization_exceeds_max_canonical_permutations_permutatio",
            "colored-graph canonicalization exceeds the "
            f"{MAX_CANONICAL_PERMUTATIONS}-permutation bound",
        )
    replay_work = canonical_replay_work(graph)
    if replay_work > MAX_CANONICAL_REPLAY_WORK:
        raise PydanticCustomError(
            "graph.colored_canonicalization_exceeds_max_canonical_replay_work",
            "colored-graph canonicalization exceeds the "
            f"{MAX_CANONICAL_REPLAY_WORK}-unit execution-and-replay work bound",
        )
    result_bytes = canonicalization_result_wire_bytes(graph)
    if result_bytes > MAX_CANONICALIZATION_RESULT_BYTES:
        raise PydanticCustomError(
            "graph.colored_canonicalization_exceeds_max_canonicalization_result_bytes",
            "colored-graph canonicalization exceeds the "
            f"{MAX_CANONICALIZATION_RESULT_BYTES}-byte result bound",
        )


__all__ = [
    "MAX_CANONICALIZATION_RESULT_BYTES",
    "canonicalization_result_wire_bytes",
    "require_admitted_colored_graph_canonicalization",
]
