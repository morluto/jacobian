"""Typed contracts for fixed-length simple path profiles."""

from __future__ import annotations

import math

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_PATH_PROFILE_SEARCH_WORK = 10_000_000
_MAX_PATH_PROFILE_ROWS = 256 * 256
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024


def _canonical_max_degree(graph: SimpleUndirectedGraph) -> int:
    degrees = dict.fromkeys(graph.vertices, 0)
    for left, right in graph.edges:
        degrees[left] += 1
        degrees[right] += 1
    return max(degrees.values(), default=0)


def _path_prefix_work_bound(vertex_count: int, max_degree: int, length: int) -> int:
    """Bound DFS nodes for one source by degree and simple-path length."""
    depth = min(length, max(vertex_count - 1, 0))
    prefixes = 1
    total = 1
    for step in range(1, depth + 1):
        prefixes *= max_degree if step == 1 else max(max_degree - 1, 0)
        total += prefixes
    return total


def _path_profile_result_bytes(graph: SimpleUndirectedGraph, path_length: int) -> int:
    vertex_count = len(graph.vertices)
    graph_bytes = len(encode_strict_json(graph.model_dump(mode="json")))
    max_label_bytes = max(
        (len(encode_strict_json(label)) for label in graph.vertices),
        default=0,
    )
    degree = _canonical_max_degree(graph)
    depth = min(path_length, max(len(graph.vertices) - 1, 0))
    max_path_count = math.prod(range(max(degree - 1, 1), max(degree - 1, 1) + depth))
    row_bytes = 2 * max_label_bytes + len(str(max_path_count)) + 32
    return (
        graph_bytes
        + vertex_count * vertex_count * row_bytes
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )


class PathProfileRequest(StrictModel):
    """A finite simple undirected graph and a path length."""

    graph: SimpleUndirectedGraph
    path_length: int = Field(ge=0, le=10)


class PathProfileRow(StrictModel):
    """One (source, target, count) triple in a path profile."""

    source: str
    target: str
    path_count: int = Field(ge=0)


class PathProfileResult(StrictModel):
    """Complete fixed-length simple path profile by endpoint."""

    source: SimpleUndirectedGraph
    path_length: int = Field(ge=0, le=10)
    rows: list[PathProfileRow] = Field(max_length=_MAX_PATH_PROFILE_ROWS)


__all__ = [
    "MAX_PATH_PROFILE_SEARCH_WORK",
    "PathProfileRequest",
    "PathProfileResult",
    "PathProfileRow",
]
