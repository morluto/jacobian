"""Cycle-length profile kernel."""

from __future__ import annotations

from collections import deque

import networkx as nx

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.cycle_length_profile._models import (
    MAX_CYCLE_LENGTH_SEARCH_WORK,
    CycleLengthEntry,
    CycleLengthProfileResult,
    _cycle_search_work,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["compute_cycle_length_profile"]


def compute_cycle_length_profile(
    graph: SimpleUndirectedGraph,
) -> CycleLengthProfileResult:
    """Return the complete set of simple-cycle lengths with witnesses.

    For every k from 3 to |V|, if the graph contains a simple k-cycle,
    include one canonical witness.
    """
    if _cycle_search_work(graph) > MAX_CYCLE_LENGTH_SEARCH_WORK:
        raise OperationDomainValidationError(
            location=("graph",),
            code="cycle_length.search_work_exceeded",
            message=(
                "cycle-length enumeration exceeds the admitted simple-path work bound"
            ),
        )
    nx_graph: nx.Graph[str] = nx.Graph()
    for v in graph.vertices:
        nx_graph.add_node(v)
    for u, v in graph.edges:
        nx_graph.add_edge(u, v)

    n = len(graph.vertices)
    entries: list[CycleLengthEntry] = []

    for target_length in range(3, n + 1):
        witness = _find_cycle_of_length(nx_graph, target_length)
        if witness is not None:
            entries.append(
                CycleLengthEntry(
                    length=target_length,
                    witness=witness,
                )
            )

    return CycleLengthProfileResult(
        graph=graph,
        entries=tuple(entries),
        cycle_lengths=tuple(e.length for e in entries),
    )


def _find_cycle_of_length(nx_graph: nx.Graph[str], k: int) -> tuple[str, ...] | None:
    """Find a simple cycle of exactly length k using BFS from each vertex."""
    for source in nx_graph.nodes:
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
        visited: set[tuple[str, ...]] = set()

        while queue:
            node, path = queue.popleft()
            if len(path) == k:
                if nx_graph.has_edge(path[-1], source):
                    return path
                continue
            if len(path) > k:
                continue
            for neighbor in sorted(nx_graph.neighbors(node)):
                if neighbor not in path:
                    new_path = (*path, neighbor)
                    if new_path not in visited:
                        visited.add(new_path)
                        queue.append((neighbor, new_path))

    return None
