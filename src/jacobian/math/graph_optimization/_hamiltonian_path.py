"""Complete bounded Hamiltonian-path decision."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graph_optimization._models import (
    GraphHamiltonianPathRequest,
    GraphHamiltonianPathResult,
)
from jacobian.math.graph_optimization._operations import build_simple_graph

if TYPE_CHECKING:
    import networkx as nx


def decide_hamiltonian_path(
    request: GraphHamiltonianPathRequest,
) -> GraphHamiltonianPathResult:
    graph = cast("nx.Graph[str]", build_simple_graph(request.graph))
    vertices = tuple(sorted(graph))
    order = len(vertices)
    if order == 0:
        return GraphHamiltonianPathResult(
            decision="EXISTS",
            order=0,
            path=(),
        )
    index = {vertex: position for position, vertex in enumerate(vertices)}
    adjacency_masks = tuple(
        sum(1 << index[neighbor] for neighbor in graph.neighbors(vertex))
        for vertex in vertices
    )
    predecessor: dict[tuple[int, int], int | None] = {
        (1 << position, position): None for position in range(order)
    }
    for mask in range(1, 1 << order):
        endings = tuple(last for last in range(order) if (mask, last) in predecessor)
        for last in endings:
            available = adjacency_masks[last] & ~mask
            while available:
                bit = available & -available
                following = bit.bit_length() - 1
                state = (mask | bit, following)
                predecessor.setdefault(state, last)
                available ^= bit
    full_mask = (1 << order) - 1
    possible_endings = tuple(
        last for last in range(order) if (full_mask, last) in predecessor
    )
    if not possible_endings:
        return GraphHamiltonianPathResult(
            decision="DOES_NOT_EXIST",
            order=order,
            path=(),
        )
    last = possible_endings[0]
    mask = full_mask
    reversed_path: list[str] = []
    while True:
        reversed_path.append(vertices[last])
        previous = predecessor[(mask, last)]
        if previous is None:
            break
        mask ^= 1 << last
        last = previous
    return GraphHamiltonianPathResult(
        decision="EXISTS",
        order=order,
        path=tuple(reversed(reversed_path)),
    )


HAMILTONIAN_PATH_OPERATION: MathTool[
    GraphHamiltonianPathRequest,
    GraphHamiltonianPathResult,
] = MathTool(
    operation_id="graph.hamiltonian_path.decide",
    version="5",
    title="Decide bounded Hamiltonian-path existence",
    description=(
        "Completely decide whether a supplied simple graph of order at most 18 "
        "has a spanning simple path. EXISTS returns the ordered path witness; "
        "DOES_NOT_EXIST follows only after exhaustive finite dynamic programming."
    ),
    request_type=GraphHamiltonianPathRequest,
    result_type=GraphHamiltonianPathResult,
    run=decide_hamiltonian_path,
    tags=(
        "graph",
        "hamiltonian-path",
        "spanning-path",
        "decision",
        "exact",
        "bounded",
    ),
    examples=(
        example(
            "path_graph_3",
            "Hamiltonian path of the path graph P3.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                },
            },
        ),
    ),
)


__all__ = ["HAMILTONIAN_PATH_OPERATION"]
