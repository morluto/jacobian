"""Typed contracts for the common-neighbour profile operation."""

import math
from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _common_neighbor_result_bytes_upper(graph: SimpleUndirectedGraph) -> int:
    """Bound the complete profile from pair and common-incidence identities."""
    vertex_count = len(graph.vertices)
    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    pair_count = math.comb(vertex_count, 2)
    pair_label_bytes = (vertex_count - 1) * sum(
        len(label.encode("utf-8")) + 3 for label in graph.vertices
    )
    common_label_bytes = sum(
        math.comb(degrees[label], 2) * (len(label.encode("utf-8")) + 3)
        for label in graph.vertices
    )
    return (
        len(encode_strict_json(graph.model_dump(mode="json")))
        + pair_count * 72
        + pair_label_bytes
        + common_label_bytes
        + vertex_count * 24
        + 512
    )


class CommonNeighborProfileRequest(StrictModel):
    """Request the common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_deliverable_profile(self) -> Self:
        if (
            _common_neighbor_result_bytes_upper(self.graph)
            > CanonicalLimits().max_output_bytes
        ):
            raise PydanticCustomError(
                "common_neighbor.output_bound",
                "common-neighbor profile exceeds the canonical output budget",
            )
        return self


class PairEntry(StrictModel):
    """One unordered vertex pair with its common-neighbour set."""

    u: str
    v: str
    common_neighbors: tuple[str, ...]
    codegree: int


class CommonNeighborProfileResult(StrictModel):
    """The complete common-neighbour profile of a graph."""

    graph: SimpleUndirectedGraph
    pairs: tuple[PairEntry, ...]
    max_codegree: int
    histogram: tuple[int, ...]
    is_c4_free: bool


__all__ = [
    "CommonNeighborProfileRequest",
    "CommonNeighborProfileResult",
    "PairEntry",
    "_common_neighbor_result_bytes_upper",
]
