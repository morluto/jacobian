from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import CanonicalLimits
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._all_terminal_reliability import (
    ALL_TERMINAL_RELIABILITY_OPERATION,
    AllTerminalReliabilityRequest,
    AllTerminalReliabilityWireResult,
    compute_all_terminal_reliability,
)
from jacobian.math.probability._graph_connection_probability import (
    GraphConnectionProbabilityRequest,
    GraphReliabilityEdgeProbability,
    compute_graph_connection_probability,
)
from jacobian.math.probability.all_terminal_reliability import (
    MAX_ALL_TERMINAL_RELIABILITY_EDGES,
    AllTerminalReliabilityResult,
    all_terminal_reliability,
)


def _graph(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _is_connected(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> bool:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {vertices[0]}
    pending = [vertices[0]]
    while pending:
        vertex = pending.pop()
        unseen = adjacency[vertex] - seen
        seen.update(unseen)
        pending.extend(unseen)
    return len(seen) == len(vertices)


def _independent_counts(graph: SimpleUndirectedGraph) -> tuple[int, ...]:
    counts = [0] * (len(graph.edges) + 1)
    for state in range(1 << len(graph.edges)):
        open_edges = tuple(
            edge for index, edge in enumerate(graph.edges) if state & (1 << index)
        )
        if _is_connected(graph.vertices, open_edges):
            counts[len(open_edges)] += 1
    return tuple(counts)


def _triangle_result() -> AllTerminalReliabilityWireResult:
    return compute_all_terminal_reliability(
        AllTerminalReliabilityRequest(
            graph=_graph(
                ("a", "b", "c"),
                (("a", "b"), ("a", "c"), ("b", "c")),
            ),
            open_probability=CanonicalRational(num="1", den="2"),
        )
    )


def test_triangle_returns_exact_coefficient_vector_and_probability() -> None:
    result = all_terminal_reliability(
        _graph(
            ("a", "b", "c"),
            (("a", "b"), ("a", "c"), ("b", "c")),
        ),
        Fraction(1, 2),
    )

    assert result.connected_spanning_subgraph_counts == (0, 0, 3, 1)
    assert result.reliability_probability == Fraction(1, 2)
    assert result.visited_states == 8


def test_rosenstock_canale_nine_vertex_profile() -> None:
    # Figure 1 and Appendix B of https://arxiv.org/abs/2212.03912.
    graph = _graph(
        tuple("abcdefghi"),
        (
            ("a", "e"),
            ("a", "f"),
            ("a", "g"),
            ("a", "h"),
            ("b", "e"),
            ("b", "f"),
            ("b", "g"),
            ("b", "h"),
            ("c", "e"),
            ("c", "g"),
            ("c", "h"),
            ("c", "i"),
            ("d", "f"),
            ("d", "g"),
            ("d", "h"),
            ("d", "i"),
            ("e", "i"),
            ("f", "i"),
        ),
    )

    result = all_terminal_reliability(graph, Fraction(1, 2))

    assert result.connected_spanning_subgraph_counts == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        12480,
        27856,
        33772,
        28344,
        17725,
        8442,
        3051,
        816,
        153,
        18,
        1,
    )
    assert result.reliability_probability == Fraction(66329, 131072)


def test_native_result_replays_its_retained_graph() -> None:
    graph = _graph(
        ("a", "b", "c"),
        (("a", "b"), ("a", "c"), ("b", "c")),
    )

    with pytest.raises(ValueError):
        AllTerminalReliabilityResult(
            graph=graph,
            open_probability=Fraction(1, 2),
            connected_spanning_subgraph_counts=(0, 0, 2, 1),
            reliability_probability=Fraction(1, 2),
            visited_states=8,
        )


def test_all_terminal_event_differs_from_two_terminal_connectivity() -> None:
    graph = _graph(
        ("a", "b", "c"),
        (("a", "b"), ("a", "c"), ("b", "c")),
    )
    half = CanonicalRational(num="1", den="2")
    two_terminal = compute_graph_connection_probability(
        GraphConnectionProbabilityRequest(
            graph=graph,
            edge_probabilities=tuple(
                GraphReliabilityEdgeProbability(edge=edge, open_probability=half)
                for edge in graph.edges
            ),
            terminals=("a", "c"),
        )
    )
    all_terminal = all_terminal_reliability(graph, Fraction(1, 2))

    assert two_terminal.connection_probability.as_fraction() == Fraction(5, 8)
    assert all_terminal.reliability_probability == Fraction(1, 2)


def test_native_kernel_matches_independent_powerset_oracle() -> None:
    probability = Fraction(2, 5)
    for vertex_count in range(1, 5):
        vertices = tuple(f"v{index}" for index in range(vertex_count))
        possible_edges = tuple(combinations(vertices, 2))
        for graph_state in range(1 << len(possible_edges)):
            graph = _graph(
                vertices,
                tuple(
                    edge
                    for index, edge in enumerate(possible_edges)
                    if graph_state & (1 << index)
                ),
            )
            expected_counts = _independent_counts(graph)
            result = all_terminal_reliability(graph, probability)
            expected_probability = sum(
                (
                    count
                    * probability**open_edges
                    * (1 - probability) ** (len(graph.edges) - open_edges)
                    for open_edges, count in enumerate(expected_counts)
                ),
                Fraction(),
            )

            assert result.connected_spanning_subgraph_counts == expected_counts
            assert result.reliability_probability == expected_probability


@pytest.mark.parametrize(
    ("graph", "probability", "counts", "reliability"),
    (
        (_graph(("a",), ()), Fraction(0), (1,), Fraction(1)),
        (
            _graph(("a", "b", "c"), (("a", "b"),)),
            Fraction(1),
            (0, 0),
            Fraction(0),
        ),
        (
            _graph(("a", "b", "c"), (("a", "b"), ("b", "c"))),
            Fraction(1, 3),
            (0, 0, 1),
            Fraction(1, 9),
        ),
    ),
    ids=("single-vertex", "isolated-vertex", "path"),
)
def test_degenerate_and_boundary_conventions(
    graph: SimpleUndirectedGraph,
    probability: Fraction,
    counts: tuple[int, ...],
    reliability: Fraction,
) -> None:
    result = all_terminal_reliability(graph, probability)

    assert result.connected_spanning_subgraph_counts == counts
    assert result.reliability_probability == reliability


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["connected_spanning_subgraph_counts"].__setitem__(
                2, "2"
            ),
            "counts do not match the source graph",
        ),
        (
            lambda value: value.__setitem__(
                "reliability_probability", {"num": "5", "den": "8"}
            ),
            "probability does not match its sources",
        ),
        (
            lambda value: value.__setitem__("visited_states", 4),
            "visited_states does not match",
        ),
        (
            lambda value: value.__setitem__(
                "open_probability", {"num": "1", "den": "3"}
            ),
            "probability does not match its sources",
        ),
        (
            lambda value: value.__setitem__(
                "graph",
                {
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                },
            ),
            "connected spanning subgraph has at least n-1 edges",
        ),
    ),
    ids=(
        "coefficient",
        "probability",
        "state-count",
        "source-probability",
        "source-graph",
    ),
)
def test_result_replay_rejects_independent_mutations(
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    candidate = deepcopy(_triangle_result().model_dump(mode="json"))
    mutation(candidate)

    with pytest.raises(ValidationError):
        AllTerminalReliabilityWireResult.model_validate(candidate)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {
                "graph": {"vertices": [], "edges": []},
                "open_probability": {"num": "1", "den": "2"},
            },
            "requires a nonempty graph",
        ),
        (
            {
                "graph": {"vertices": ["a"], "edges": []},
                "open_probability": {"num": "2", "den": "1"},
            },
            r"must lie in \[0, 1\]",
        ),
        (
            {
                "graph": {"vertices": ["a"], "edges": []},
                "open_probability": {"num": "1", "den": "1" + "0" * 128},
            },
            "128-digit bound",
        ),
        (
            {
                "graph": {
                    "vertices": [f"v{index:02d}" for index in range(7)],
                    "edges": [
                        list(edge)
                        for edge in combinations(
                            tuple(f"v{index:02d}" for index in range(7)), 2
                        )
                    ],
                },
                "open_probability": {"num": "1", "den": "2"},
            },
            "edge bound",
        ),
    ),
    ids=("null", "probability", "digits", "edges"),
)
def test_request_rejects_outside_complete_domain(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError):
        AllTerminalReliabilityRequest.model_validate(payload)


def test_operation_executes_the_twenty_edge_enumeration_boundary() -> None:
    vertices = tuple(f"v{index:02d}" for index in range(21))
    request = AllTerminalReliabilityRequest(
        graph=_graph(
            vertices,
            tuple((vertices[index], vertices[index + 1]) for index in range(20)),
        ),
        open_probability=CanonicalRational(num="1", den="2"),
    )

    result = compute_all_terminal_reliability(request)

    assert result.connected_spanning_subgraph_counts == (
        *("0" for _ in range(MAX_ALL_TERMINAL_RELIABILITY_EDGES)),
        "1",
    )
    assert result.reliability_probability.as_fraction() == Fraction(1, 1 << 20)
    assert result.visited_states == 1 << 20


def test_request_allows_more_vertices_when_the_state_space_is_small() -> None:
    graph = _graph(tuple(f"v{index:03d}" for index in range(256)), ())

    request = AllTerminalReliabilityRequest(
        graph=graph,
        open_probability=CanonicalRational(num="1", den="2"),
    )
    result = compute_all_terminal_reliability(request)

    assert len(request.graph.vertices) == 256
    assert result.connected_spanning_subgraph_counts == ("0",)
    assert result.reliability_probability.as_fraction() == 0


def test_native_boundary_rejects_an_oversized_retained_graph() -> None:
    graph = _graph(("x" * CanonicalLimits().max_output_bytes,), ())

    with pytest.raises(ValueError):
        all_terminal_reliability(graph, Fraction(1, 2))


def test_native_boundary_rejects_huge_probability_components_preflight() -> None:
    graph = _graph(("v",), ())
    at_limit = all_terminal_reliability(graph, Fraction(1, 10**127))

    assert at_limit.reliability_probability == 1
    with pytest.raises(ValueError):
        all_terminal_reliability(graph, Fraction(1, 10**200_000 + 3))


def test_request_schema_exposes_the_retained_result_limit() -> None:
    schema = AllTerminalReliabilityRequest.model_json_schema()

    assert (
        "retained graph plus the fixed exact-result envelope"
        in str(schema["description"]).lower()
    )
    assert (
        "retained graph plus fixed result headroom"
        in str(schema["properties"]["graph"]["description"]).lower()
    )


def test_operation_declares_and_executes_copyable_example() -> None:
    operation = ALL_TERMINAL_RELIABILITY_OPERATION
    payload = operation.examples[0].input
    request = operation.request_type.model_validate(payload)
    result = operation.run(request)

    assert operation.operation_id == (
        "probability.graph_reliability.all_terminal.compute"
    )
    assert result.event == "ALL_VERTICES_CONNECTED"
    assert result.reliability_probability.as_fraction() == Fraction(1, 2)
