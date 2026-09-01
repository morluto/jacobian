"""Typed contracts for uniform-subset intersection graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Literal

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_GRAPH_LABEL_BYTES,
    MAX_INDEXED_SIMPLE_GRAPH_EDGES,
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

MAX_SAFE_GROUND_SET_SIZE = (1 << 53) - 1

IntersectionRelation = Literal["INTERSECTION_LT_THRESHOLD", "INTERSECTION_EQ_THRESHOLD"]


@dataclass(frozen=True)
class _UniformSubsetIntersectionPlan:
    """Request-scoped exact size quantities used by the constructor."""

    vertex_count: int
    edge_count: int


def _combination_at_most(n: int, k: int, limit: int) -> int:
    """Return ``C(n, k)`` exactly up to ``limit``, then return ``limit + 1``.

    The recurrence is checked before each multiplication, so a rejected
    family never constructs an unrestricted large binomial coefficient.
    """

    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for index in range(1, k + 1):
        numerator = n - k + index
        if value * numerator > limit * index:
            return limit + 1
        value = value * numerator // index
    return value


def _intersection_pair_count(n: int, k: int, t: int, relation: str) -> int:
    qualifying = (
        (size for size in range(k + 1) if size < t)
        if relation == "INTERSECTION_LT_THRESHOLD"
        else (size for size in range(k + 1) if size == t)
    )
    neighbors_per_vertex = sum(
        comb(k, size) * comb(n - k, k - size)
        for size in qualifying
        if k - size <= n - k
    )
    if relation == "INTERSECTION_EQ_THRESHOLD" and t == k:
        neighbors_per_vertex -= 1
    return comb(n, k) * neighbors_per_vertex // 2


def _largest_subset_label_bytes(n: int, k: int) -> int:
    if k == 0:
        return 2
    if 2 * k + 1 > MAX_GRAPH_LABEL_BYTES:
        return 2 * k + 1
    return 2 + (k - 1) + sum(len(str(value)) for value in range(n - k, n))


def _admit_uniform_subset_intersection(
    ground_set_size: int,
    subset_cardinality: int,
    threshold: int,
    relation: IntersectionRelation,
) -> _UniformSubsetIntersectionPlan:
    """Validate one request and derive its exact materialization sizes."""

    for name, value in (
        ("ground_set_size", ground_set_size),
        ("subset_cardinality", subset_cardinality),
        ("threshold", threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if ground_set_size > MAX_SAFE_GROUND_SET_SIZE:
        raise ValueError(f"ground_set_size must not exceed {MAX_SAFE_GROUND_SET_SIZE}")
    if subset_cardinality > ground_set_size:
        raise ValueError("subset_cardinality must not exceed ground_set_size")
    if threshold > subset_cardinality:
        raise ValueError("threshold must not exceed subset_cardinality")
    if relation not in (
        "INTERSECTION_LT_THRESHOLD",
        "INTERSECTION_EQ_THRESHOLD",
    ):
        raise ValueError("relation must name a supported intersection predicate")

    vertex_count = _combination_at_most(
        ground_set_size,
        subset_cardinality,
        MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    )
    if vertex_count > MAX_INDEXED_SIMPLE_GRAPH_VERTICES:
        raise ValueError(
            "the uniform subset family exceeds the "
            f"{MAX_INDEXED_SIMPLE_GRAPH_VERTICES}-vertex graph bound"
        )
    if (
        _largest_subset_label_bytes(ground_set_size, subset_cardinality)
        > MAX_GRAPH_LABEL_BYTES
    ):
        raise ValueError(
            "a canonical subset label exceeds the "
            f"{MAX_GRAPH_LABEL_BYTES}-byte graph-label bound"
        )
    edge_count = _intersection_pair_count(
        ground_set_size, subset_cardinality, threshold, relation
    )
    if edge_count > MAX_INDEXED_SIMPLE_GRAPH_EDGES:
        raise ValueError(
            "the selected intersection relation exceeds the "
            f"{MAX_INDEXED_SIMPLE_GRAPH_EDGES}-edge graph bound"
        )
    return _UniformSubsetIntersectionPlan(
        vertex_count=vertex_count,
        edge_count=edge_count,
    )


class UniformSubsetIntersectionRequest(StrictModel):
    """Construct a graph from k-subsets of [n] with a threshold relation."""

    ground_set_size: StrictInt = Field(ge=0, le=MAX_SAFE_GROUND_SET_SIZE)
    subset_cardinality: StrictInt = Field(ge=0, le=MAX_SAFE_GROUND_SET_SIZE)
    threshold: StrictInt = Field(ge=0, le=MAX_SAFE_GROUND_SET_SIZE)
    relation: IntersectionRelation


class UniformSubsetIntersectionResult(StrictModel):
    """The constructed uniform-subset intersection graph."""

    ground_set_size: StrictInt
    subset_cardinality: StrictInt
    threshold: StrictInt
    relation: IntersectionRelation
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_SAFE_GROUND_SET_SIZE",
    "IntersectionRelation",
    "UniformSubsetIntersectionRequest",
    "UniformSubsetIntersectionResult",
]
