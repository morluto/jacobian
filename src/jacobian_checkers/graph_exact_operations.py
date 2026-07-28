"""Independent finite-graph replay with no producer or numeric-backend imports."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _run(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay_method: str,
    replay: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject(
                f"declared result does not match independent {replay_method}"
            )
        return _accept(f"independent {replay_method} accepted {operation_id}")
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def _finite_simple_graph(
    source: dict[str, Any],
    *,
    maximum_order: int,
) -> tuple[tuple[str, ...], set[tuple[str, str]], dict[str, set[str]]]:
    graph = source.get("graph")
    if not isinstance(graph, dict) or set(graph) != {
        "graph_schema_version",
        "vertices",
        "edges",
    }:
        raise ValueError("graph input is malformed")
    vertices = graph["vertices"]
    edges = graph["edges"]
    if (
        graph["graph_schema_version"] != "1"
        or not isinstance(vertices, list)
        or len(vertices) > maximum_order
        or not all(
            isinstance(vertex, str) and 0 < len(vertex) <= 256 for vertex in vertices
        )
        or len(vertices) != len(set(vertices))
        or not isinstance(edges, list)
        or len(edges) > len(vertices) * (len(vertices) - 1) // 2
    ):
        raise ValueError("graph lies outside the checker scope")
    vertex_set = set(vertices)
    normalized_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] == edge[1]
            or edge[0] not in vertex_set
            or edge[1] not in vertex_set
        ):
            raise ValueError("graph edge payload is malformed")
        normalized_edges.add(tuple(sorted((edge[0], edge[1]))))
    if len(normalized_edges) != len(edges):
        raise ValueError("graph edge payload contains duplicates")
    adjacency = {vertex: set[str]() for vertex in vertices}
    for left, right in normalized_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(vertices), normalized_edges, adjacency


def _induced_tree_maximum(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=16,
    )
    vertex_set = set(vertices)

    def is_induced_tree(candidate: tuple[str, ...]) -> bool:
        if not candidate:
            return False
        selected = set(candidate)
        edge_count = sum(
            1
            for left, right in normalized_edges
            if left in selected and right in selected
        )
        if edge_count != len(candidate) - 1:
            return False
        reached = {candidate[0]}
        frontier = [candidate[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] & selected:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return len(reached) == len(candidate)

    claimed = result.get("optimum_value")
    witness = result.get("witness_vertices")
    if (
        result.get("status") != "EXACT"
        or result.get("convention") != "NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO"
        or type(claimed) is not int
        or claimed < 0
        or claimed > len(vertices)
        or result.get("order") != len(vertices)
        or result.get("incumbent_value") != claimed
        or result.get("lower_bound") != claimed
        or result.get("upper_bound") != claimed
        or not isinstance(witness, list)
        or len(witness) != claimed
        or not all(isinstance(vertex, str) for vertex in witness)
        or len(witness) != len(set(witness))
        or any(vertex not in vertex_set for vertex in witness)
    ):
        return False
    if claimed == 0:
        if vertices or witness:
            return False
    elif not is_induced_tree(tuple(witness)):
        return False

    actual = 0
    for cardinality in range(len(vertices), 0, -1):
        if any(
            is_induced_tree(candidate)
            for candidate in combinations(vertices, cardinality)
        ):
            actual = cardinality
            break
    return actual == claimed


def check_graph_induced_tree_maximum(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.induced_tree.maximum.compute",
        witness_format="graph.induced-tree.maximum.exhaustive-replay",
        replay=_induced_tree_maximum,
        replay_method="finite-subset exhaustive replay",
    )


def _maximum_matching(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=32,
    )
    if set(result) != {
        "maximum_matching_cardinality",
        "witness_edges",
        "certificate",
    }:
        return False
    claimed = result["maximum_matching_cardinality"]
    witness = result["witness_edges"]
    if (
        type(claimed) is not int
        or claimed < 0
        or claimed > len(vertices) // 2
        or not isinstance(witness, list)
        or len(witness) != claimed
    ):
        return False
    parsed_witness: list[tuple[str, str]] = []
    used_vertices: set[str] = set()
    for edge in witness:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] >= edge[1]
            or (edge[0], edge[1]) not in normalized_edges
            or edge[0] in used_vertices
            or edge[1] in used_vertices
        ):
            return False
        parsed_witness.append((edge[0], edge[1]))
        used_vertices.update(edge)
    if parsed_witness != sorted(parsed_witness):
        return False

    certificate = result["certificate"]
    if not isinstance(certificate, dict) or set(certificate) != {
        "certificate_schema_version",
        "kind",
        "barrier_vertices",
        "odd_component_count",
        "upper_bound",
    }:
        return False
    barrier = certificate["barrier_vertices"]
    declared_odd_components = certificate["odd_component_count"]
    declared_upper_bound = certificate["upper_bound"]
    vertex_set = set(vertices)
    if (
        certificate["certificate_schema_version"] != "1"
        or certificate["kind"] != "TUTTE_BERGE_BARRIER"
        or not isinstance(barrier, list)
        or not all(isinstance(vertex, str) for vertex in barrier)
        or barrier != sorted(set(barrier))
        or any(vertex not in vertex_set for vertex in barrier)
        or type(declared_odd_components) is not int
        or type(declared_upper_bound) is not int
        or declared_odd_components < 0
        or declared_upper_bound < 0
    ):
        return False

    barrier_set = set(barrier)
    unseen = vertex_set - barrier_set
    odd_component_count = 0
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        component = {first}
        frontier = [first]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - barrier_set:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        odd_component_count += len(component) % 2

    numerator = len(vertices) + len(barrier) - odd_component_count
    if numerator < 0 or numerator % 2:
        return False
    upper_bound = numerator // 2
    return (
        declared_odd_components == odd_component_count
        and declared_upper_bound == upper_bound
        and claimed == upper_bound
    )


def check_graph_maximum_matching(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.invariant.maximum_matching.compute",
        witness_format="graph.maximum-matching.tutte-berge-v1",
        replay=_maximum_matching,
        replay_method="Tutte-Berge barrier replay",
    )


__all__ = [
    "check_graph_induced_tree_maximum",
    "check_graph_maximum_matching",
]
