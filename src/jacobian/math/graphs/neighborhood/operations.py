"""Exact open-neighbourhood kernel for a selected vertex set."""

from __future__ import annotations

from jacobian.math.graphs.neighborhood._bounds import (
    OpenNeighborhoodAdmission,
    admit_open_neighborhood,
)
from jacobian.math.graphs.neighborhood._models import NeighborhoodResult
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["open_neighborhood"]


def open_neighborhood(
    graph: SimpleUndirectedGraph,
    selected_vertices: tuple[str, ...],
) -> NeighborhoodResult:
    """Return the exact open neighbourhood N_G(S).

    The open neighbourhood of a vertex set S consists of all vertices
    outside S that are adjacent to at least one member of S.
    """
    return _open_neighborhood_from_admission(
        graph, admit_open_neighborhood(graph, selected_vertices)
    )


def _open_neighborhood_from_admission(
    graph: SimpleUndirectedGraph,
    admission: OpenNeighborhoodAdmission,
) -> NeighborhoodResult:
    """Build the result from one already-computed mathematical admission."""
    return NeighborhoodResult(
        graph=graph,
        selected_vertices=admission.selected_vertices,
        neighborhood=admission.neighborhood,
    )
