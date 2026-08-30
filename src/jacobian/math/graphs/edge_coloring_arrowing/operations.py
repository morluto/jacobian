"""Edge-colouring Ramsey arrowing decision for finite graphs."""

from __future__ import annotations

from itertools import permutations, product

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring._models import EdgeColoringAssignment
from jacobian.math.graphs.edge_coloring_arrowing._models import (
    EdgeColoringArrowingResult,
    _validate_arrowing_envelope,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["decide_edge_coloring_arrowing"]


def decide_edge_coloring_arrowing(
    host_graph: SimpleUndirectedGraph,
    targets: tuple[SimpleUndirectedGraph, ...],
) -> EdgeColoringArrowingResult:
    """Decide whether host_graph arrows (T_1, ..., T_q) under edge colouring.

    The host graph H arrows (T_1,...,T_q) if every q-edge-colouring of H
    contains a monochromatic copy of T_i in colour i for some i.

    Returns ARROWS if every colouring contains a monochromatic target copy,
    or DOES_NOT_ARROW with one avoiding colouring otherwise.
    """
    try:
        _validate_arrowing_envelope(host_graph, targets)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error

    edges = list(host_graph.edges)
    num_colors = len(targets)
    edge_count = len(edges)

    for coloring in product(range(num_colors), repeat=edge_count):
        if _is_avoiding_coloring(coloring, edges, targets, host_graph):
            avoiding = EdgeColoringAssignment(
                graph=host_graph,
                colors=num_colors,
                coloring=coloring,
            )
            return EdgeColoringArrowingResult(
                host_graph=host_graph,
                targets=targets,
                outcome="DOES_NOT_ARROW",
                avoiding_coloring=avoiding,
            )

    return EdgeColoringArrowingResult(
        host_graph=host_graph,
        targets=targets,
        outcome="ARROWS",
    )


def _is_avoiding_coloring(
    coloring: tuple[int, ...],
    edges: list[tuple[str, str]],
    targets: tuple[SimpleUndirectedGraph, ...],
    host: SimpleUndirectedGraph,
) -> bool:
    """Check if a colouring avoids all monochromatic target copies."""
    for color, target in enumerate(targets):
        color_edges = [edges[i] for i in range(len(edges)) if coloring[i] == color]
        if _contains_target(host.vertices, color_edges, target):
            return False
    return True


def _contains_target(
    host_vertices: tuple[str, ...],
    colored_edges: list[tuple[str, str]],
    target: SimpleUndirectedGraph,
) -> bool:
    """Check if the target graph appears as a subgraph using only colored_edges."""
    colored_edge_set = set(colored_edges)
    host_vertex_list = list(host_vertices)
    target_vertices = list(target.vertices)

    if len(target_vertices) > len(host_vertex_list):
        return False

    for vertex_assignment in permutations(host_vertex_list, len(target_vertices)):
        vmap = dict(zip(target_vertices, vertex_assignment, strict=True))
        found = True
        for a, _b in target.edges:
            ha, hb = vmap[a], vmap[_b]
            edge = (min(ha, hb), max(ha, hb))
            if edge not in colored_edge_set:
                found = False
                break
        if found:
            return True
    return False
