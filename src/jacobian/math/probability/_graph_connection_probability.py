"""Exact bounded undirected terminal-reliability operation contract."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import (
    canonicalize_json,
    format_canonical_integer,
)
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._models import MAX_INPUT_RATIONAL_DIGITS

MAX_GRAPH_RELIABILITY_VERTICES = 16
MAX_GRAPH_RELIABILITY_EDGES = 12
MAX_GRAPH_RELIABILITY_STATES = 1 << MAX_GRAPH_RELIABILITY_EDGES
MAX_GRAPH_RELIABILITY_LEDGER_BYTES = 9 * 1024 * 1024
GRAPH_RELIABILITY_LEDGER_FIXED_BYTES = 16 * 1024


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.model_invariant", message)


class GraphReliabilityEdgeProbability(StrictModel):
    edge: tuple[str, str]
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_bounded_probability(self) -> Self:
        if len(self.edge) != 2 or self.edge[0] >= self.edge[1]:
            raise _validation_error(
                "reliability edge must contain two ordered vertices"
            )
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="graph reliability edge probability",
        )
        return self


class GraphConnectionProbabilityRequest(StrictModel):
    graph: SimpleUndirectedGraph
    edge_probabilities: tuple[GraphReliabilityEdgeProbability, ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"


class GraphReliabilityState(StrictModel):
    state_index: StrictInt = Field(ge=0, lt=MAX_GRAPH_RELIABILITY_STATES)
    open_edges: tuple[tuple[str, str], ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals_connected: bool
    state_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_probability(self) -> Self:
        if not 0 <= self.state_probability.as_fraction() <= 1:
            raise _validation_error(
                "graph reliability state probability must lie in [0, 1]"
            )
        return self


class GraphConnectionProbabilityResult(StrictModel):
    terminals: tuple[str, str]
    connection_probability: CanonicalRational
    edge_count: StrictInt = Field(ge=0, le=MAX_GRAPH_RELIABILITY_EDGES)
    visited_states: StrictInt = Field(ge=1, le=MAX_GRAPH_RELIABILITY_STATES)
    states: tuple[GraphReliabilityState, ...] = Field(
        min_length=1,
        max_length=MAX_GRAPH_RELIABILITY_STATES,
    )
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"
    edge_independence: Literal["INDEPENDENT_BERNOULLI"] = "INDEPENDENT_BERNOULLI"
    enumeration: Literal["COMPLETE_EDGE_SUBSETS"] = "COMPLETE_EDGE_SUBSETS"

    @model_validator(mode="after")
    def require_canonical_state_ledger(self) -> Self:
        if self.visited_states != 1 << self.edge_count:
            raise _validation_error("visited state count is not the full edge powerset")
        if len(self.states) != self.visited_states:
            raise _validation_error("state ledger length does not match visited states")
        if tuple(item.state_index for item in self.states) != tuple(
            range(self.visited_states)
        ):
            raise _validation_error(
                "state ledger indices must be complete and canonical"
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build the complete ledger after the powerset kernel finishes."""

        return cls.model_construct(**values)


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.p)),
        den=format_canonical_integer(int(value.q)),
    )


def _admit_graph_connection_request(
    request: GraphConnectionProbabilityRequest,
) -> tuple[tuple[Any, ...], int]:
    """Normalize edge probabilities and admit the complete ledger envelope."""
    if len(request.graph.vertices) > MAX_GRAPH_RELIABILITY_VERTICES:
        raise ValueError(
            "graph reliability exceeds the "
            f"{MAX_GRAPH_RELIABILITY_VERTICES}-vertex bound"
        )
    if len(request.graph.edges) > MAX_GRAPH_RELIABILITY_EDGES:
        raise ValueError(
            f"graph reliability exceeds the {MAX_GRAPH_RELIABILITY_EDGES}-edge bound"
        )
    if tuple(item.edge for item in request.edge_probabilities) != request.graph.edges:
        raise ValueError(
            "edge probabilities must cover graph edges in canonical graph order"
        )
    if (
        len(request.terminals) != 2
        or request.terminals[0] == request.terminals[1]
        or any(terminal not in request.graph.vertices for terminal in request.terminals)
    ):
        raise ValueError("terminals must be two distinct declared graph vertices")

    from flint import fmpq

    probabilities = []
    for item in request.edge_probabilities:
        probability = item.open_probability.as_fraction()
        probabilities.append(fmpq(probability.numerator, probability.denominator))
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("graph reliability probabilities must lie in [0, 1]")

    edge_count = len(request.graph.edges)
    state_count = 1 << edge_count
    repeated_edge_bytes = (
        (1 << (edge_count - 1))
        * sum(len(canonicalize_json(list(edge))) + 1 for edge in request.graph.edges)
        if edge_count
        else 0
    )
    probability_numerator_digits = sum(
        max(
            len(
                format_canonical_integer(item.open_probability.as_fraction().numerator)
            ),
            len(
                format_canonical_integer(
                    (1 - item.open_probability.as_fraction()).numerator
                )
            ),
        )
        for item in request.edge_probabilities
    )
    probability_denominator_digits = sum(
        len(format_canonical_integer(item.open_probability.as_fraction().denominator))
        for item in request.edge_probabilities
    )
    maximum_state = {
        "state_index": state_count - 1,
        "open_edges": [],
        "terminals_connected": False,
        "state_probability": {
            "num": "9" * max(1, probability_numerator_digits),
            "den": "9" * max(1, probability_denominator_digits),
        },
    }
    estimated_ledger_bytes = (
        repeated_edge_bytes
        + state_count * len(canonicalize_json(maximum_state))
        + GRAPH_RELIABILITY_LEDGER_FIXED_BYTES
    )
    if estimated_ledger_bytes > MAX_GRAPH_RELIABILITY_LEDGER_BYTES:
        raise ValueError(
            "graph reliability request can exceed the complete ledger "
            f"budget of {MAX_GRAPH_RELIABILITY_LEDGER_BYTES} bytes"
        )
    return tuple(probabilities), state_count


def _terminals_connected(
    vertices: tuple[str, ...],
    open_edges: tuple[tuple[str, str], ...],
    terminals: tuple[str, str],
) -> bool:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for left, right in open_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {terminals[0]}
    pending = [terminals[0]]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex] - seen:
            if neighbor == terminals[1]:
                return True
            seen.add(neighbor)
            pending.append(neighbor)
    return terminals[1] in seen


def compute_graph_connection_probability(
    request: GraphConnectionProbabilityRequest,
) -> GraphConnectionProbabilityResult:
    """Compute exact terminal connectivity over every undirected edge subset."""

    from flint import fmpq

    probabilities, state_count = _admit_graph_connection_request(request)
    states: list[GraphReliabilityState] = []
    connection_probability = fmpq(0)
    for state_index in range(state_count):
        open_edges = tuple(
            edge
            for index, edge in enumerate(request.graph.edges)
            if state_index & (1 << index)
        )
        state_probability = fmpq(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        connected = _terminals_connected(
            request.graph.vertices,
            open_edges,
            request.terminals,
        )
        if connected:
            connection_probability += state_probability
        states.append(
            GraphReliabilityState(
                state_index=state_index,
                open_edges=open_edges,
                terminals_connected=connected,
                state_probability=_wire(state_probability),
            )
        )
    return GraphConnectionProbabilityResult._from_kernel(
        terminals=request.terminals,
        connection_probability=_wire(connection_probability),
        edge_count=len(request.graph.edges),
        visited_states=state_count,
        states=tuple(states),
    )


_SQUARE_GRAPH = {
    "graph": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"]],
    },
    "edge_probabilities": [
        {"edge": ["a", "b"], "open_probability": {"num": "1", "den": "2"}},
        {"edge": ["a", "c"], "open_probability": {"num": "1", "den": "2"}},
        {"edge": ["b", "d"], "open_probability": {"num": "1", "den": "2"}},
        {"edge": ["c", "d"], "open_probability": {"num": "1", "den": "2"}},
    ],
    "terminals": ["a", "d"],
}


GRAPH_CONNECTION_PROBABILITY_OPERATION = MathTool(
    operation_id="probability.graph_reliability.connection_probability.compute",
    title="Exact small-graph terminal connection probability",
    description=(
        "Compute the exact probability that two explicit terminals are connected "
        "in one bounded undirected graph with independent rational edge-open "
        "probabilities, preserving the complete edge-subset ledger."
    ),
    request_type=GraphConnectionProbabilityRequest,
    result_type=GraphConnectionProbabilityResult,
    run=compute_graph_connection_probability,
    tags=(
        "probability",
        "graph",
        "reliability",
        "percolation",
        "connection",
        "terminals",
        "exact",
        "bounded",
        "python-flint",
    ),
    examples=(
        example(
            "triangle_terminal_reliability",
            "Compute the exact terminal connection probability in a fair-edge triangle.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                },
                "edge_probabilities": [
                    {
                        "edge": ["a", "b"],
                        "open_probability": {"num": "1", "den": "2"},
                    },
                    {
                        "edge": ["a", "c"],
                        "open_probability": {"num": "1", "den": "2"},
                    },
                    {
                        "edge": ["b", "c"],
                        "open_probability": {"num": "1", "den": "2"},
                    },
                ],
                "terminals": ["a", "c"],
            },
        ),
        example(
            "square_terminal_reliability",
            "Compute square-graph terminal reliability; edge probabilities cover edges canonically and terminals are distinct declared vertices.",
            _SQUARE_GRAPH,
        ),
    ),
)


__all__ = [
    "GRAPH_CONNECTION_PROBABILITY_OPERATION",
    "MAX_GRAPH_RELIABILITY_EDGES",
    "MAX_GRAPH_RELIABILITY_STATES",
    "MAX_GRAPH_RELIABILITY_VERTICES",
    "GraphConnectionProbabilityRequest",
    "GraphConnectionProbabilityResult",
    "GraphReliabilityEdgeProbability",
    "GraphReliabilityState",
    "compute_graph_connection_probability",
]
