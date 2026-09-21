"""Exact Hamiltonian-path dynamic-programming regressions."""

from itertools import permutations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.optimization._hamiltonian_path import decide_hamiltonian_path
from jacobian.math.graphs.optimization._models import GraphHamiltonianPathRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _request(order: int, edges: tuple[tuple[str, str], ...]) -> GraphHamiltonianPathRequest:
    vertices = tuple(str(index) for index in range(order))
    return GraphHamiltonianPathRequest(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges)
    )


def _brute_force_exists(request: GraphHamiltonianPathRequest) -> bool:
    graph = request.graph
    edge_set = {frozenset(edge) for edge in graph.edges}
    return any(
        all(frozenset((order[index], order[index + 1])) in edge_set for index in range(len(order) - 1))
        for order in permutations(graph.vertices)
    )


def test_bitset_dp_matches_exhaustive_oracle_on_every_graph_through_order_four() -> None:
    for order in range(5):
        vertices = tuple(str(index) for index in range(order))
        possible = tuple(
            (vertices[left], vertices[right])
            for left in range(order)
            for right in range(left + 1, order)
        )
        for mask in range(1 << len(possible)):
            request = _request(
                order,
                tuple(edge for index, edge in enumerate(possible) if mask & (1 << index)),
            )
            result = decide_hamiltonian_path(request)
            expected = _brute_force_exists(request)
            assert (result.decision == "EXISTS") is expected
            if expected:
                assert len(result.path) == order
                assert set(result.path) == set(request.graph.vertices)
                edge_set = {frozenset(edge) for edge in request.graph.edges}
                assert all(
                    frozenset((result.path[index], result.path[index + 1])) in edge_set
                    for index in range(max(0, order - 1))
                )


def test_order_bound_remains_a_typed_domain_error() -> None:
    with pytest.raises(OperationDomainValidationError):
        decide_hamiltonian_path(_request(19, ()))
