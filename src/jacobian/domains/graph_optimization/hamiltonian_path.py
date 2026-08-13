"""Complete bounded Hamiltonian-path decision."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from jacobian.contracts.graph_optimization import (
    GraphHamiltonianPathRequest,
    GraphHamiltonianPathResult,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.domains.graph_optimization.operations import build_simple_graph
from jacobian.operation_bindings import inline_operation
from jacobian.operation_declarations import OperationDeclaration
from jacobian.operations import (
    OperationRefusalError,
)

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


def _execute(
    request: GraphHamiltonianPathRequest,
) -> GraphHamiltonianPathResult:
    import networkx as nx

    try:
        return decide_hamiltonian_path(request)
    except (
        ArithmeticError,
        nx.NetworkXError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OperationRefusalError(
            OperationDiagnostic(
                code="GRAPH_HAMILTONIAN_PATH_NOT_APPLICABLE",
                stage="graph_hamiltonian_path_computation",
                message=str(exc),
                hint=(
                    "Supply a finite simple graph of order at most 18; larger "
                    "graphs are outside this complete decision scope."
                ),
            )
        ) from exc


HAMILTONIAN_PATH_OPERATION: OperationDeclaration[
    GraphHamiltonianPathRequest,
    GraphHamiltonianPathResult,
] = inline_operation(
    OperationDeclaration(
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
        execute=_execute,
        tags=(
            "graph",
            "hamiltonian-path",
            "spanning-path",
            "decision",
            "exact",
            "bounded",
        ),
    )
)


__all__ = ["HAMILTONIAN_PATH_OPERATION"]
