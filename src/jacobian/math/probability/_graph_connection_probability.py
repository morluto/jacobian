"""Exact bounded undirected terminal-reliability operation contract."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability._models import MAX_INPUT_RATIONAL_DIGITS

MAX_GRAPH_RELIABILITY_VERTICES = 16
MAX_GRAPH_RELIABILITY_EDGES = 12
MAX_GRAPH_RELIABILITY_STATES = 1 << MAX_GRAPH_RELIABILITY_EDGES


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
    """Transport request for terminal reliability."""

    graph: SimpleUndirectedGraph
    edge_probabilities: tuple[GraphReliabilityEdgeProbability, ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"


class GraphReliabilitySource(StrictModel):
    """Canonical graph, edge-axis probabilities, and terminal event source."""

    graph: SimpleUndirectedGraph
    edge_probabilities: tuple[GraphReliabilityEdgeProbability, ...] = Field(
        max_length=MAX_GRAPH_RELIABILITY_EDGES
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_CONNECTED"] = "TERMINALS_CONNECTED"

    @model_validator(mode="after")
    def require_bound_edge_axis(self) -> Self:
        if tuple(item.edge for item in self.edge_probabilities) != self.graph.edges:
            raise _validation_error(
                "reliability source edge probabilities must follow the graph edge axis"
            )
        if (
            len(self.terminals) != 2
            or self.terminals[0] == self.terminals[1]
            or any(vertex not in self.graph.vertices for vertex in self.terminals)
        ):
            raise _validation_error(
                "reliability source terminals must be graph vertices"
            )
        if len(self.graph.edges) > MAX_GRAPH_RELIABILITY_EDGES:
            raise _validation_error("reliability source exceeds the edge bound")
        return self


class GraphReliabilityState(StrictModel):
    state_index: StrictInt = Field(ge=0, lt=MAX_GRAPH_RELIABILITY_STATES)
    open_edge_indices: tuple[StrictInt, ...] = Field(
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

    @model_validator(mode="after")
    def require_canonical_edge_indices(self) -> Self:
        if any(index < 0 for index in self.open_edge_indices):
            raise _validation_error(
                "reliability state edge indices must be nonnegative"
            )
        if self.open_edge_indices != tuple(sorted(set(self.open_edge_indices))):
            raise _validation_error("reliability state edge indices must be canonical")
        return self


class GraphConnectionProbabilityResult(StrictModel):
    source: GraphReliabilitySource
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
        if not 0 <= self.connection_probability.as_fraction() <= 1:
            raise _validation_error(
                "reliability connection probability must lie in [0, 1]"
            )
        if self.terminals != self.source.terminals or self.event != self.source.event:
            raise _validation_error("reliability result must retain its source event")
        if self.edge_count != len(self.source.graph.edges):
            raise _validation_error("reliability result edge axis mismatch")
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
        if any(
            any(index >= self.edge_count for index in state.open_edge_indices)
            for state in self.states
        ):
            raise _validation_error("reliability state edge axis mismatch")
        if any(
            state.state_index != sum(1 << index for index in state.open_edge_indices)
            for state in self.states
        ):
            raise _validation_error("reliability state index does not encode its edges")
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        """Build the complete ledger after the powerset kernel finishes."""

        return cls.model_construct(**values)


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=int(value.p),
        den=int(value.q),
    )


def _admit_graph_connection_request(
    request: GraphConnectionProbabilityRequest | GraphReliabilitySource,
) -> tuple[tuple[Any, ...], int]:
    """Normalize edge probabilities and admit the complete ledger envelope."""
    if len(request.graph.vertices) > MAX_GRAPH_RELIABILITY_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="probability.graph_reliability.vertex_bound",
            message=(
                "graph reliability exceeds the "
                f"{MAX_GRAPH_RELIABILITY_VERTICES}-vertex bound"
            ),
        )
    if len(request.graph.edges) > MAX_GRAPH_RELIABILITY_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="probability.graph_reliability.edge_bound",
            message=(
                f"graph reliability exceeds the {MAX_GRAPH_RELIABILITY_EDGES}-edge bound"
            ),
        )
    if tuple(item.edge for item in request.edge_probabilities) != request.graph.edges:
        raise OperationDomainValidationError(
            location=("edge_probabilities",),
            code="probability.graph_reliability.edge_probability_binding",
            message=(
                "edge probabilities must cover graph edges in canonical graph order"
            ),
        )
    if (
        len(request.terminals) != 2
        or request.terminals[0] == request.terminals[1]
        or any(terminal not in request.graph.vertices for terminal in request.terminals)
    ):
        raise OperationDomainValidationError(
            location=("terminals",),
            code="probability.graph_reliability.terminals",
            message="terminals must be two distinct declared graph vertices",
        )

    from flint import fmpq

    probabilities = []
    for item in request.edge_probabilities:
        probability = item.open_probability.as_fraction()
        probabilities.append(fmpq(probability.numerator, probability.denominator))
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise OperationDomainValidationError(
            location=("edge_probabilities",),
            code="probability.graph_reliability.probability_range",
            message="graph reliability probabilities must lie in [0, 1]",
        )

    edge_count = len(request.graph.edges)
    state_count = 1 << edge_count
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
    request: GraphReliabilitySource,
) -> GraphConnectionProbabilityResult:
    """Compute exact terminal connectivity over every undirected edge subset."""

    probabilities, state_count = _admit_graph_connection_request(request)
    source = GraphReliabilitySource.model_validate(request.model_dump())
    return _compute_graph_connection_probability_admitted(
        source, probabilities, state_count
    )


def _compute_graph_connection_probability_admitted(
    source: GraphReliabilitySource,
    probabilities: tuple[Any, ...],
    state_count: int,
) -> GraphConnectionProbabilityResult:
    """Compute an already admitted reliability ledger."""

    from flint import fmpq

    edge_count = len(source.graph.edges)
    states: list[GraphReliabilityState] = []
    connection_probability = fmpq(0)
    for state_index in range(state_count):
        open_edge_indices = tuple(
            index for index in range(edge_count) if state_index & (1 << index)
        )
        open_edges = tuple(source.graph.edges[index] for index in open_edge_indices)
        state_probability = fmpq(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        connected = _terminals_connected(
            source.graph.vertices,
            open_edges,
            source.terminals,
        )
        if connected:
            connection_probability += state_probability
        states.append(
            GraphReliabilityState(
                state_index=state_index,
                open_edge_indices=open_edge_indices,
                terminals_connected=connected,
                state_probability=_wire(state_probability),
            )
        )
    return GraphConnectionProbabilityResult._from_kernel(
        source=source,
        terminals=source.terminals,
        connection_probability=_wire(connection_probability),
        edge_count=len(source.graph.edges),
        visited_states=state_count,
        states=tuple(states),
    )


def verify_graph_connection_probability(
    claim: GraphConnectionProbabilityResult,
) -> bool:
    """Verify the reliability ledger and aggregate claim against its source."""

    try:
        expected = compute_graph_connection_probability(claim.source)
        return expected == claim
    except (TypeError, ValueError):
        return False


def _compute_graph_connection_probability_request(
    request: GraphConnectionProbabilityRequest,
) -> GraphConnectionProbabilityResult:
    """Project a transport request into the domain-owned computation."""

    probabilities, state_count = _admit_graph_connection_request(request)
    source = GraphReliabilitySource.model_validate(request.model_dump())
    return _compute_graph_connection_probability_admitted(
        source, probabilities, state_count
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
    run=_compute_graph_connection_probability_request,
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
        OperationExample(
            name="triangle_terminal_reliability",
            description="Compute the exact terminal connection probability in a fair-edge triangle.",
            input={
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
        OperationExample(
            name="square_terminal_reliability",
            description="Compute square-graph terminal reliability; edge probabilities cover edges canonically and terminals are distinct declared vertices.",
            input=_SQUARE_GRAPH,
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
    "GraphReliabilitySource",
    "GraphReliabilityState",
    "compute_graph_connection_probability",
    "verify_graph_connection_probability",
]
