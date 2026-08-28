"""Exact induced vertex-subset counting with the pinned NetworkX 3.6.1 backend."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.patterns._models import (
    _require_bounded_request,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _integer_graph(value: SimpleUndirectedGraph) -> nx.Graph[int]:
    """Convert canonical labels once so enumeration work uses bounded integers."""

    indices = {vertex: index for index, vertex in enumerate(value.vertices)}
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(range(len(value.vertices)))
    graph.add_edges_from((indices[left], indices[right]) for left, right in value.edges)
    return graph


def _explicit_induced_candidate(
    host: nx.Graph[int],
    subset: tuple[int, ...],
) -> nx.Graph[int]:
    """Materialize one induced candidate with work depending only on its order."""

    candidate: nx.Graph[int] = nx.Graph()
    candidate.add_nodes_from(range(len(subset)))
    candidate.add_edges_from(
        (left_index, right_index)
        for left_index, right_index in combinations(range(len(subset)), 2)
        if host.has_edge(subset[left_index], subset[right_index])
    )
    return candidate


def count_induced_vertex_subset_patterns(
    host: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
) -> int:
    """Return the exact number of host vertex subsets inducing ``pattern``."""

    host_order = len(host.vertices)
    pattern_order = len(pattern.vertices)
    if pattern_order > host_order:
        return 0
    if pattern_order == 0:
        return 1

    host_graph = _integer_graph(host)
    pattern_graph = _integer_graph(pattern)
    pattern_edge_count = pattern_graph.number_of_edges()
    pattern_degrees = tuple(sorted(degree for _, degree in pattern_graph.degree()))

    occurrence_count = 0
    for subset in combinations(range(host_order), pattern_order):
        induced = _explicit_induced_candidate(host_graph, subset)
        if induced.number_of_edges() != pattern_edge_count:
            continue
        if tuple(sorted(degree for _, degree in induced.degree())) != pattern_degrees:
            continue
        if nx.vf2pp_is_isomorphic(induced, pattern_graph):
            occurrence_count += 1
    return occurrence_count


def induced_vertex_subset_pattern_count(
    host: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
) -> int:
    """Return the exact number of vertex subsets of ``host`` inducing ``pattern``."""
    try:
        _require_bounded_request(host, pattern)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("host", "pattern"),
            code=exc.type,
            message=exc.message(),
        ) from None
    return count_induced_vertex_subset_patterns(host, pattern)


__all__ = ["induced_vertex_subset_pattern_count"]
