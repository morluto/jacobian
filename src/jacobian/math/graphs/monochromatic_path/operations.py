"""Monochromatic path hypergraph constructor."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.graphs.monochromatic_path._models import (
    MAX_VERTICES,
    MonochromaticPathResult,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph

__all__ = ["construct_monochromatic_path_hypergraphs"]

MAX_MONOCHROMATIC_PATH_WORK = 100_000_000


def _json_array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _admit_monochromatic_path_graph(graph: ColoredUndirectedGraph) -> tuple[str, ...]:
    vertex_count = len(graph.graph.vertices)
    edge_count = len(graph.graph.edges)
    if vertex_count > MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.monochromatic_path.vertex_count_exceeds_bound",
            message=f"monochromatic path search supports at most {MAX_VERTICES} vertices",
        )
    if edge_count and not graph.edge_colors:
        raise OperationDomainValidationError(
            location=("graph", "edge_colors"),
            code="graph.monochromatic_path.edge_colors_must_cover_edges",
            message="edge_colors must assign one color to every graph edge",
        )
    if len(graph.edge_colors) not in (0, edge_count):
        raise OperationDomainValidationError(
            location=("graph", "edge_colors"),
            code="graph.monochromatic_path.edge_colors_must_cover_edges",
            message="edge_colors must be empty or align with every graph edge",
        )

    colours = tuple(sorted(set(graph.edge_colors))) or ("uncolored",)
    subset_count = (1 << vertex_count) - 1
    work = len(colours) * max(1, vertex_count * vertex_count * (1 << vertex_count))
    if work > MAX_MONOCHROMATIC_PATH_WORK:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.monochromatic_path.work_exceeds_bound",
            message="monochromatic path search exceeds its exact work bound",
        )
    # Each colour produces its own hypergraph.  The finite-hypergraph edge
    # bound applies per result, while the aggregate output is bounded below.
    edge_upper_bound = subset_count
    if edge_upper_bound > MAX_EDGES:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.monochromatic_path.edge_count_exceeds_bound",
            message="complete monochromatic path hypergraphs exceed the edge bound",
        )
    incidence_upper_bound = vertex_count * (1 << (vertex_count - 1))
    if incidence_upper_bound > MAX_TOTAL_INCIDENCES:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.monochromatic_path.incidence_count_exceeds_bound",
            message="complete monochromatic path hypergraphs exceed the incidence bound",
        )

    return colours


def construct_monochromatic_path_hypergraphs(
    graph: ColoredUndirectedGraph,
) -> MonochromaticPathResult:
    """For each colour, return the hypergraph whose edges are vertex sets
    that admit a monochromatic simple path using only edges of that colour.

    Singletons are included (length-0 path convention).
    """
    colours = _admit_monochromatic_path_graph(graph)
    vertices = list(graph.graph.vertices)
    edges = list(graph.graph.edges)
    edge_colors = list(graph.edge_colors)

    colour_to_adjacency: dict[str, dict[str, set[str]]] = {
        c: {v: set() for v in vertices} for c in colours
    }
    for (a, b), c in zip(edges, edge_colors, strict=True):
        colour_to_adjacency[c][a].add(b)
        colour_to_adjacency[c][b].add(a)

    result: dict[str, FiniteHypergraph] = {}

    for colour in colours:
        adj = colour_to_adjacency[colour]
        supports = _hamiltonian_supports(tuple(vertices), adj)

        hyper_edges: list[tuple[str, tuple[str, ...]]] = []
        for i, support in enumerate(sorted(supports)):
            hyper_edges.append((f"path_{i}", support))

        result[colour] = FiniteHypergraph(
            vertices=tuple(vertices),
            edges=tuple(hyper_edges),
        )

    return MonochromaticPathResult(
        graph=graph,
        colour_to_hypergraph=result,
    )


def _has_hamiltonian_path(
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> bool:
    """Check if the subgraph induced on `vertices` has a Hamiltonian path."""
    return tuple(vertices) in _hamiltonian_supports(vertices, adjacency)


def _hamiltonian_supports(
    vertices: tuple[str, ...], adjacency: dict[str, set[str]]
) -> tuple[tuple[str, ...], ...]:
    """Return all supports admitting a simple path using subset DP."""
    n = len(vertices)
    index = {vertex: position for position, vertex in enumerate(vertices)}
    neighbor_masks = [
        sum(1 << index[neighbor] for neighbor in adjacency[vertex])
        for vertex in vertices
    ]
    endpoint_masks = [0] * (1 << n)
    for vertex in range(n):
        endpoint_masks[1 << vertex] = 1 << vertex
    for mask in range(1, 1 << n):
        endpoints = endpoint_masks[mask]
        while endpoints:
            endpoint_bit = endpoints & -endpoints
            endpoints -= endpoint_bit
            endpoint = endpoint_bit.bit_length() - 1
            additions = neighbor_masks[endpoint] & ~mask
            while additions:
                addition = additions & -additions
                additions -= addition
                endpoint_masks[mask | addition] |= addition
    return tuple(
        tuple(vertices[position] for position in range(n) if mask & (1 << position))
        for mask in range(1, 1 << n)
        if endpoint_masks[mask]
    )
