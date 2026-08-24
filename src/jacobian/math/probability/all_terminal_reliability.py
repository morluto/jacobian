"""Exact all-terminal reliability for bounded simple graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import TYPE_CHECKING, Literal

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

if TYPE_CHECKING:
    import networkx as nx

MAX_ALL_TERMINAL_RELIABILITY_EDGES = 20
MAX_ALL_TERMINAL_RELIABILITY_STATES = 1 << MAX_ALL_TERMINAL_RELIABILITY_EDGES
MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS = 128
# For p=a/b, the reduced reliability denominator divides b**m and its
# numerator is no larger because the result is a probability.
MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS = (
    MAX_ALL_TERMINAL_RELIABILITY_EDGES * MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS + 1
)
_MAX_ALL_TERMINAL_RELIABILITY_INPUT_ABS = 10**MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS
_MAX_ALL_TERMINAL_RELIABILITY_RESULT_ABS = (
    10**MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS
)
# The largest non-graph fields contain two 128-digit probability components,
# two 2,561-digit result components, 21 seven-digit coefficients, and fixed
# JSON field names. Eight KiB leaves conservative headroom for that payload.
_RESULT_ENVELOPE_RESERVE_BYTES = 8_192


def _require_bounded_problem(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
) -> None:
    if not isinstance(graph, SimpleUndirectedGraph):
        raise TypeError("graph must be a SimpleUndirectedGraph")
    vertex_count = len(graph.vertices)
    edge_count = len(graph.edges)
    if vertex_count == 0:
        raise ValueError("all-terminal reliability requires a nonempty graph")
    if edge_count > MAX_ALL_TERMINAL_RELIABILITY_EDGES:
        raise ValueError(
            "all-terminal reliability exceeds the "
            f"{MAX_ALL_TERMINAL_RELIABILITY_EDGES}-edge bound"
        )
    if type(open_probability) is not Fraction:
        raise TypeError("open_probability must be a Fraction")
    if not 0 <= open_probability <= 1:
        raise ValueError("open_probability must lie in [0, 1]")
    if (
        abs(open_probability.numerator) >= _MAX_ALL_TERMINAL_RELIABILITY_INPUT_ABS
        or open_probability.denominator >= _MAX_ALL_TERMINAL_RELIABILITY_INPUT_ABS
    ):
        raise ValueError(
            "open_probability exceeds the "
            f"{MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS}-digit bound"
        )

    output_limit = CanonicalLimits().max_output_bytes
    output_error = (
        "the all-terminal reliability result retains its source graph and "
        f"would exceed the {output_limit}-byte canonical output limit; "
        "shorten vertex labels"
    )
    # Every label code point occupies at least one output byte. This cheap lower
    # bound rejects arbitrarily large native values before RFC 8785 expansion.
    # Once it passes, JSON escaping can expand by at most a fixed factor within
    # the canonical output limit, so the exact serialization below is bounded.
    retained_label_code_points = sum(len(vertex) for vertex in graph.vertices) + sum(
        len(left) + len(right) for left, right in graph.edges
    )
    if retained_label_code_points + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
        raise ValueError(output_error)
    try:
        graph_bytes = len(encode_strict_json(graph.model_dump(mode="json")))
    except ValueError as exc:
        raise ValueError(output_error) from exc
    if graph_bytes + _RESULT_ENVELOPE_RESERVE_BYTES > output_limit:
        raise ValueError(output_error)


def _indexed_graph(
    graph: SimpleUndirectedGraph,
) -> tuple[nx.Graph[int], tuple[tuple[int, int], ...]]:
    import networkx as nx

    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    indexed_edges = tuple(
        (vertex_index[left], vertex_index[right]) for left, right in graph.edges
    )
    backend_graph: nx.Graph[int] = nx.Graph()
    backend_graph.add_nodes_from(range(len(graph.vertices)))
    return backend_graph, indexed_edges


def _connected_spanning_subgraph_counts(
    graph: SimpleUndirectedGraph,
) -> tuple[int, ...]:
    """Count connected spanning edge subsets, indexed by subset cardinality."""

    import networkx as nx

    backend_graph, edges = _indexed_graph(graph)
    counts = [0] * (len(edges) + 1)

    previous_gray_code = 0
    open_edge_count = 0
    for state_index in range(1 << len(edges)):
        gray_code = state_index ^ (state_index >> 1)
        if state_index:
            changed_bit = gray_code ^ previous_gray_code
            edge_index = changed_bit.bit_length() - 1
            edge = edges[edge_index]
            if gray_code & changed_bit:
                backend_graph.add_edge(*edge)
                open_edge_count += 1
            else:
                backend_graph.remove_edge(*edge)
                open_edge_count -= 1
        if open_edge_count >= len(graph.vertices) - 1 and nx.is_connected(
            backend_graph
        ):
            counts[open_edge_count] += 1
        previous_gray_code = gray_code
    return tuple(counts)


def _evaluate_reliability(
    counts: tuple[int, ...],
    open_probability: Fraction,
) -> Fraction:
    closed_probability = 1 - open_probability
    edge_count = len(counts) - 1
    return sum(
        (
            count
            * open_probability**open_edges
            * closed_probability ** (edge_count - open_edges)
            for open_edges, count in enumerate(counts)
        ),
        Fraction(),
    )


def _require_source_bound_result(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
    counts: tuple[int, ...],
    reliability_probability: Fraction,
    visited_states: int,
) -> None:
    _require_bounded_problem(graph, open_probability)
    if (
        abs(reliability_probability.numerator)
        >= _MAX_ALL_TERMINAL_RELIABILITY_RESULT_ABS
        or reliability_probability.denominator
        >= _MAX_ALL_TERMINAL_RELIABILITY_RESULT_ABS
    ):
        raise ValueError(
            "all-terminal reliability result probability exceeds the "
            f"{MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS}-digit bound"
        )
    edge_count = len(graph.edges)
    if len(counts) != edge_count + 1:
        raise ValueError(
            "connected-spanning-subgraph counts must cover edge counts 0..m"
        )
    for open_edges, count in enumerate(counts):
        if not 0 <= count <= comb(edge_count, open_edges):
            raise ValueError(
                "connected-spanning-subgraph count lies outside its subset class"
            )
        if open_edges < len(graph.vertices) - 1 and count:
            raise ValueError("a connected spanning subgraph has at least n-1 edges")

    expected_counts = _connected_spanning_subgraph_counts(graph)
    if counts != expected_counts:
        raise ValueError(
            "connected-spanning-subgraph counts do not match the source graph"
        )
    expected_probability = _evaluate_reliability(expected_counts, open_probability)
    if reliability_probability != expected_probability:
        raise ValueError(
            "all-terminal reliability probability does not match its sources"
        )
    if visited_states != 1 << edge_count:
        raise ValueError("visited_states does not match the complete edge powerset")


@dataclass(frozen=True, slots=True)
class AllTerminalReliabilityResult:
    """A graph-bound exact reliability value and its coefficient profile."""

    graph: SimpleUndirectedGraph
    open_probability: Fraction
    connected_spanning_subgraph_counts: tuple[int, ...]
    reliability_probability: Fraction
    visited_states: int
    event: Literal["ALL_VERTICES_CONNECTED"] = "ALL_VERTICES_CONNECTED"

    def __post_init__(self) -> None:
        if type(self.connected_spanning_subgraph_counts) is not tuple or any(
            type(count) is not int for count in self.connected_spanning_subgraph_counts
        ):
            raise TypeError(
                "connected-spanning-subgraph counts must be a tuple of integers"
            )
        if type(self.reliability_probability) is not Fraction:
            raise TypeError("reliability_probability must be a Fraction")
        if type(self.visited_states) is not int:
            raise TypeError("visited_states must be an integer")
        if self.event != "ALL_VERTICES_CONNECTED":
            raise ValueError("unsupported all-terminal reliability event")
        _require_source_bound_result(
            self.graph,
            self.open_probability,
            self.connected_spanning_subgraph_counts,
            self.reliability_probability,
            self.visited_states,
        )


def _compute_all_terminal_reliability(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
) -> tuple[tuple[int, ...], Fraction, int]:
    _require_bounded_problem(graph, open_probability)
    counts = _connected_spanning_subgraph_counts(graph)
    return (
        counts,
        _evaluate_reliability(counts, open_probability),
        1 << len(graph.edges),
    )


def all_terminal_reliability(
    graph: SimpleUndirectedGraph,
    open_probability: Fraction,
) -> AllTerminalReliabilityResult:
    """Compute exact uniform-edge all-terminal reliability.

    The coefficient at index ``k`` counts spanning connected subgraphs with
    exactly ``k`` open edges. The probability is
    ``sum(c[k] * p**k * (1-p)**(m-k))``.
    """

    counts, reliability_probability, visited_states = _compute_all_terminal_reliability(
        graph, open_probability
    )
    return AllTerminalReliabilityResult(
        graph=graph,
        open_probability=open_probability,
        connected_spanning_subgraph_counts=counts,
        reliability_probability=reliability_probability,
        visited_states=visited_states,
    )


__all__ = [
    "AllTerminalReliabilityResult",
    "all_terminal_reliability",
]
