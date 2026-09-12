"""Exact bounded site-reliability operation for a finite simple graph.

Site percolation differs from the existing bond operation in what is random:
every vertex is independently open with its declared probability, and the two
terminals connect when both are open and lie in one connected component of the
open induced subgraph.  The state space is the vertex powerset, so this is a
separate public contract rather than a mode flag on the bond operation.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, StrictInt, ValidationError, model_validator
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

MAX_SITE_RELIABILITY_VERTICES = 12
MAX_SITE_RELIABILITY_EDGES = 24
MAX_SITE_RELIABILITY_STATES = 1 << MAX_SITE_RELIABILITY_VERTICES
# A state mass contains one probability (or its complement) per vertex.  The
# input contract allows 128 digits per factor, and summing at most 2**n state
# masses adds at most log10(2**n) digits.  Keep the result carrier wide enough for the
# admitted arithmetic; admission below uses the request's actual factor
# heights, so ordinary small requests are not penalised by this worst case.
MAX_SITE_RELIABILITY_RATIONAL_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * MAX_SITE_RELIABILITY_VERTICES
    + MAX_SITE_RELIABILITY_VERTICES
)
MAX_SITE_RELIABILITY_OUTPUT_BYTES = 64_000_000


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.model_invariant", message)


class SiteReliabilityVertexProbability(StrictModel):
    vertex: str
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_bounded_probability(self) -> Self:
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="site reliability vertex probability",
        )
        return self


class GraphSiteReliabilitySource(StrictModel):
    """Canonical graph, vertex-axis probabilities, and terminal event source."""

    graph: SimpleUndirectedGraph
    vertex_probabilities: tuple[SiteReliabilityVertexProbability, ...] = Field(
        max_length=MAX_SITE_RELIABILITY_VERTICES
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_OPEN_AND_CONNECTED"] = "TERMINALS_OPEN_AND_CONNECTED"

    @model_validator(mode="after")
    def require_bound_vertex_axis(self) -> Self:
        if tuple(item.vertex for item in self.vertex_probabilities) != tuple(
            sorted(self.graph.vertices)
        ):
            raise _validation_error(
                "site reliability vertex probabilities must follow the declared "
                "canonical vertex axis"
            )
        if (
            len(self.terminals) != 2
            or self.terminals[0] == self.terminals[1]
            or any(terminal not in self.graph.vertices for terminal in self.terminals)
        ):
            raise _validation_error(
                "site reliability terminals must be two distinct graph vertices"
            )
        if len(self.graph.vertices) > MAX_SITE_RELIABILITY_VERTICES:
            raise _validation_error("site reliability source exceeds the vertex bound")
        return self


class GraphSiteReliabilityState(StrictModel):
    """One exact open-vertex state of the site-percolation powerset."""

    state_index: StrictInt = Field(ge=0, lt=MAX_SITE_RELIABILITY_STATES)
    open_vertices: tuple[str, ...] = Field(max_length=MAX_SITE_RELIABILITY_VERTICES)
    terminals_connected: bool
    state_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_probability(self) -> Self:
        require_bounded_rational(
            self.state_probability,
            max_digits=MAX_SITE_RELIABILITY_RATIONAL_DIGITS,
            label="site reliability state probability",
        )
        if not 0 <= self.state_probability.as_fraction() <= 1:
            raise _validation_error(
                "site reliability state probability must lie in [0, 1]"
            )
        if self.open_vertices != tuple(sorted(set(self.open_vertices))):
            raise _validation_error(
                "site reliability state vertex lists must be canonical"
            )
        return self


class GraphSiteReliabilityResult(StrictModel):
    """An exact complete site-reliability result with its state ledger."""

    source: GraphSiteReliabilitySource
    connection_probability: CanonicalRational
    vertex_count: StrictInt = Field(ge=0, le=MAX_SITE_RELIABILITY_VERTICES)
    visited_states: StrictInt = Field(ge=1, le=MAX_SITE_RELIABILITY_STATES)
    states: tuple[GraphSiteReliabilityState, ...] = Field(
        min_length=1, max_length=MAX_SITE_RELIABILITY_STATES
    )
    event: Literal["TERMINALS_OPEN_AND_CONNECTED"] = "TERMINALS_OPEN_AND_CONNECTED"
    vertex_independence: Literal["INDEPENDENT_BERNOULLI"] = "INDEPENDENT_BERNOULLI"
    enumeration: Literal["COMPLETE_VERTEX_SUBSETS"] = "COMPLETE_VERTEX_SUBSETS"

    @model_validator(mode="after")
    def require_canonical_state_ledger(self) -> Self:
        require_bounded_rational(
            self.connection_probability,
            max_digits=MAX_SITE_RELIABILITY_RATIONAL_DIGITS,
            label="site connection probability",
        )
        if not 0 <= self.connection_probability.as_fraction() <= 1:
            raise _validation_error(
                "site reliability connection probability must lie in [0, 1]"
            )
        if self.vertex_count != len(self.source.graph.vertices):
            raise _validation_error("site reliability vertex axis mismatch")
        if self.visited_states != 1 << self.vertex_count:
            raise _validation_error(
                "visited state count is not the full vertex powerset"
            )
        if len(self.states) != self.visited_states:
            raise _validation_error("state ledger length does not match visited states")
        if tuple(state.state_index for state in self.states) != tuple(
            range(self.visited_states)
        ):
            raise _validation_error(
                "state ledger indices must be complete and canonical"
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(num=int(value.p), den=int(value.q))


def _open_terminals_connected(
    *,
    open_vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    terminals: tuple[str, str],
) -> bool:
    """Both terminals open and joined inside the open induced subgraph."""

    if terminals[0] not in open_vertices or terminals[1] not in open_vertices:
        return False
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in open_vertices}
    for left, right in edges:
        if left in adjacency and right in adjacency:
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


def _admit_site_request(
    request: GraphSiteReliabilitySource,
) -> None:
    if len(request.graph.vertices) > MAX_SITE_RELIABILITY_VERTICES:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="probability.site_reliability.vertex_bound",
            message=(
                "site reliability exceeds the "
                f"{MAX_SITE_RELIABILITY_VERTICES}-vertex bound"
            ),
        )
    if len(request.graph.edges) > MAX_SITE_RELIABILITY_EDGES:
        raise OperationDomainValidationError(
            location=("graph", "edges"),
            code="probability.site_reliability.edge_bound",
            message=(
                f"site reliability exceeds the {MAX_SITE_RELIABILITY_EDGES}-edge bound"
            ),
        )
    if tuple(item.vertex for item in request.vertex_probabilities) != tuple(
        sorted(request.graph.vertices)
    ):
        raise OperationDomainValidationError(
            location=("vertex_probabilities",),
            code="probability.site_reliability.vertex_probability_binding",
            message=(
                "vertex probabilities must cover graph vertices in canonical order"
            ),
        )
    if any(
        not 0 <= item.open_probability.as_fraction() <= 1
        for item in request.vertex_probabilities
    ):
        raise OperationDomainValidationError(
            location=("vertex_probabilities",),
            code="probability.site_reliability.probability_range",
            message="site reliability probabilities must lie in [0, 1]",
        )

    # Bound both the exact numerator/denominator height and the complete
    # ledger before entering the powerset loop.  Complements have numerator
    # ``den-num``; taking the maximum over both branches is a sound bound for
    # every state.  The state sum can add at most bit_length(state_count)
    # decimal digits.
    factor_digits = sum(
        max(
            len(str(abs(item.open_probability.as_fraction().numerator))),
            len(str(item.open_probability.as_fraction().denominator)),
            len(
                str(
                    abs(
                        item.open_probability.as_fraction().denominator
                        - item.open_probability.as_fraction().numerator
                    )
                )
            ),
        )
        for item in request.vertex_probabilities
    )
    state_count = 1 << len(request.graph.vertices)
    rational_digits = factor_digits + len(str(state_count))
    if rational_digits > MAX_SITE_RELIABILITY_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("vertex_probabilities",),
            code="probability.site_reliability.rational_height_bound",
            message=(
                "site reliability state masses exceed the admitted exact "
                f"{MAX_SITE_RELIABILITY_RATIONAL_DIGITS}-digit result bound"
            ),
        )
    label_bytes = sum(len(vertex.encode("utf-8")) for vertex in request.graph.vertices)
    max_component_bytes = max(
        (len(vertex.encode("utf-8")) for vertex in request.graph.vertices),
        default=1,
    )
    # This over-allocates every state to its largest possible open-vertex
    # list, which is intentionally conservative and independent of execution.
    estimated_output_bytes = (
        1024
        + label_bytes * 8
        + len(request.vertex_probabilities) * (2 * MAX_INPUT_RATIONAL_DIGITS + 96)
        + state_count
        * (
            192
            + len(request.graph.vertices) * max_component_bytes
            + 2 * rational_digits
        )
    )
    if estimated_output_bytes > MAX_SITE_RELIABILITY_OUTPUT_BYTES:
        raise OperationDomainValidationError(
            location=("states",),
            code="probability.site_reliability.output_bound",
            message="complete site-reliability ledger exceeds its output bound",
        )


def compute_site_connection_probability(
    request: GraphSiteReliabilitySource,
) -> GraphSiteReliabilityResult:
    """Compute exact terminal connectivity over every open vertex subset."""

    from flint import fmpq

    try:
        source = GraphSiteReliabilitySource.model_validate(request.model_dump())
    except ValidationError as exc:
        error = exc.errors()[0]
        raise OperationDomainValidationError(
            location=tuple(error["loc"]),
            code=str(error["type"]),
            message=str(error["msg"]),
        ) from None
    _admit_site_request(source)
    vertices = tuple(sorted(source.graph.vertices))
    probabilities = tuple(
        fmpq(
            item.open_probability.as_fraction().numerator,
            item.open_probability.as_fraction().denominator,
        )
        for item in source.vertex_probabilities
    )
    states: list[GraphSiteReliabilityState] = []
    connection_probability = fmpq(0)
    for state_index in range(1 << len(vertices)):
        open_vertices = tuple(
            vertex
            for index, vertex in enumerate(vertices)
            if state_index & (1 << index)
        )
        state_probability = fmpq(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        connected = _open_terminals_connected(
            open_vertices=open_vertices,
            edges=source.graph.edges,
            terminals=source.terminals,
        )
        if connected:
            connection_probability += state_probability
        states.append(
            GraphSiteReliabilityState(
                state_index=state_index,
                open_vertices=open_vertices,
                terminals_connected=connected,
                state_probability=_wire(state_probability),
            )
        )
    return GraphSiteReliabilityResult._from_kernel(
        source=source,
        connection_probability=_wire(connection_probability),
        vertex_count=len(vertices),
        visited_states=len(states),
        states=tuple(states),
    )


SITE_CONNECTION_PROBABILITY_OPERATION = MathTool(
    operation_id="probability.graph_site_reliability.connection_probability.compute",
    title="Compute exact terminal site-reliability of a finite graph",
    description=(
        "Given a bounded finite simple graph, one exact independent open "
        "probability for every vertex, and two distinct terminals, enumerate "
        "every vertex subset and return the exact probability that both "
        "terminals are open and lie in one connected component of the open "
        "induced subgraph, together with the complete canonical state ledger."
    ),
    request_type=GraphSiteReliabilitySource,
    result_type=GraphSiteReliabilityResult,
    run=compute_site_connection_probability,
    tags=("probability", "reliability", "site-percolation", "exact"),
    examples=(
        OperationExample(
            name="path_of_two",
            description=(
                "Two vertices joined by an edge connect exactly when both are "
                "open, so the probability is p^2 = 1/4."
            ),
            input={
                "graph": {"vertices": ["a", "b"], "edges": [["a", "b"]]},
                "vertex_probabilities": [
                    {"vertex": "a", "open_probability": {"num": "1", "den": "2"}},
                    {"vertex": "b", "open_probability": {"num": "1", "den": "2"}},
                ],
                "terminals": ["a", "b"],
            },
        ),
    ),
)

__all__ = [
    "MAX_SITE_RELIABILITY_EDGES",
    "MAX_SITE_RELIABILITY_OUTPUT_BYTES",
    "MAX_SITE_RELIABILITY_STATES",
    "MAX_SITE_RELIABILITY_VERTICES",
    "SITE_CONNECTION_PROBABILITY_OPERATION",
    "GraphSiteReliabilityResult",
    "GraphSiteReliabilitySource",
    "GraphSiteReliabilityState",
    "SiteReliabilityVertexProbability",
    "compute_site_connection_probability",
]
