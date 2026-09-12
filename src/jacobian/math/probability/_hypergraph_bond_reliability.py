"""Exact bounded hyperedge bond reliability on a finite simple hypergraph.

Two vertices connect in the open hypergraph when they lie in one connected
component of the incidence graph restricted to open hyperedges: a chain of open
hyperedges whose successive members intersect.  This is the incidence-graph
convention, not the clique-expansion convention, and the difference is stated
and tested explicitly.
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
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.probability._models import MAX_INPUT_RATIONAL_DIGITS

MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES = 12
MAX_HYPERGRAPH_RELIABILITY_STATES = 1 << MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES
# A state mass has one probability (or complement) factor per hyperedge and
# the successful-state sum has at most 2**k terms.  This is the result-carrier
# width for the admitted 128-digit input factors; request-specific admission
# below computes a tighter bound for each source.
MAX_HYPERGRAPH_RELIABILITY_RATIONAL_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES
    + MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES
)
# One unit reserves either a retained scalar digit, label code point, container
# slot, or fixed record field.  Concrete transports own encoded-byte ceilings.
MAX_HYPERGRAPH_RELIABILITY_LEDGER_UNITS = 64_000_000
# This calibrated total work budget charges actual traversal vertices and
# incidences per state below.  Declared isolated vertices are retained once in
# the source/result and are charged by the separate source/output envelope.
MAX_HYPERGRAPH_RELIABILITY_LOGICAL_WORK = 5_373_952


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.model_invariant", message)


class HyperedgeOpenProbability(StrictModel):
    hyperedge_id: str
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_canonical_bounded_probability(self) -> Self:
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="hypergraph reliability hyperedge probability",
        )
        return self


class HypergraphBondReliabilitySource(StrictModel):
    """Canonical hypergraph, hyperedge-axis probabilities, and terminals."""

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Finite simple hypergraph with at most "
            f"{MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES} nonempty hyperedges "
            "for complete hyperedge-subset enumeration. Traversal work is "
            "charged by relevant hyperedge incidences, while declared vertices "
            "are retained and charged separately."
        )
    )
    # The shared finite-hypergraph carrier is larger than this operation's
    # powerset envelope.  Keep this structural model broad and let operation
    # admission own the resource rejection.
    hyperedge_probabilities: tuple[HyperedgeOpenProbability, ...] = Field(
        description=(
            "One independent exact open probability for every hyperedge, "
            "in the hypergraph's declared hyperedge-axis order."
        )
    )
    terminals: tuple[str, str]
    event: Literal["TERMINALS_CONNECTED_IN_INCIDENCE_GRAPH"] = (
        "TERMINALS_CONNECTED_IN_INCIDENCE_GRAPH"
    )

    @model_validator(mode="after")
    def require_bound_hyperedge_axis(self) -> Self:
        if tuple(item.hyperedge_id for item in self.hyperedge_probabilities) != tuple(
            edge_id for edge_id, _ in self.hypergraph.edges
        ):
            raise _validation_error(
                "hypergraph reliability probabilities must follow the declared "
                "hyperedge axis"
            )
        if (
            len(self.terminals) != 2
            or self.terminals[0] == self.terminals[1]
            or any(
                terminal not in self.hypergraph.vertices for terminal in self.terminals
            )
        ):
            raise _validation_error(
                "hypergraph reliability terminals must be two distinct declared vertices"
            )
        members = tuple(members for _, members in self.hypergraph.edges)
        if any(not edge_members for edge_members in members):
            raise _validation_error(
                "hypergraph reliability hyperedges must be nonempty"
            )
        if len(set(members)) != len(members):
            raise _validation_error(
                "hypergraph reliability hyperedges must have distinct vertex sets"
            )
        return self


class HypergraphBondReliabilityState(StrictModel):
    state_index: StrictInt = Field(ge=0, lt=MAX_HYPERGRAPH_RELIABILITY_STATES)
    open_hyperedge_ids: tuple[str, ...] = Field(
        max_length=MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES
    )
    terminals_connected: bool
    state_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_probability(self) -> Self:
        require_bounded_rational(
            self.state_probability,
            max_digits=MAX_HYPERGRAPH_RELIABILITY_RATIONAL_DIGITS,
            label="hypergraph reliability state probability",
        )
        if not 0 <= self.state_probability.as_fraction() <= 1:
            raise _validation_error(
                "hypergraph reliability state probability must lie in [0, 1]"
            )
        if len(set(self.open_hyperedge_ids)) != len(self.open_hyperedge_ids):
            raise _validation_error("hypergraph state hyperedge IDs must be canonical")
        return self


class HypergraphBondConnectionProbabilityResult(StrictModel):
    source: HypergraphBondReliabilitySource
    connection_probability: CanonicalRational
    hyperedge_count: StrictInt = Field(ge=0, le=MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES)
    visited_states: StrictInt = Field(ge=1, le=MAX_HYPERGRAPH_RELIABILITY_STATES)
    states: tuple[HypergraphBondReliabilityState, ...] = Field(
        min_length=1, max_length=MAX_HYPERGRAPH_RELIABILITY_STATES
    )
    event: Literal["TERMINALS_CONNECTED_IN_INCIDENCE_GRAPH"] = (
        "TERMINALS_CONNECTED_IN_INCIDENCE_GRAPH"
    )
    hyperedge_independence: Literal["INDEPENDENT_BERNOULLI"] = "INDEPENDENT_BERNOULLI"
    connectivity_convention: Literal["INCIDENCE_GRAPH_CHAIN"] = "INCIDENCE_GRAPH_CHAIN"
    enumeration: Literal["COMPLETE_HYPEREDGE_SUBSETS"] = "COMPLETE_HYPEREDGE_SUBSETS"

    @model_validator(mode="after")
    def require_canonical_state_ledger(self) -> Self:
        require_bounded_rational(
            self.connection_probability,
            max_digits=MAX_HYPERGRAPH_RELIABILITY_RATIONAL_DIGITS,
            label="hypergraph connection probability",
        )
        if not 0 <= self.connection_probability.as_fraction() <= 1:
            raise _validation_error(
                "hypergraph connection probability must lie in [0, 1]"
            )
        if self.hyperedge_count != len(self.source.hypergraph.edges):
            raise _validation_error("hypergraph reliability hyperedge axis mismatch")
        if self.visited_states != 1 << self.hyperedge_count:
            raise _validation_error(
                "visited state count is not the full hyperedge powerset"
            )
        if len(self.states) != self.visited_states:
            raise _validation_error("state ledger length does not match visited states")
        if tuple(state.state_index for state in self.states) != tuple(
            range(self.visited_states)
        ):
            raise _validation_error(
                "state ledger indices must be complete and canonical"
            )
        hyperedge_axis = tuple(edge_id for edge_id, _ in self.source.hypergraph.edges)
        for state in self.states:
            expected = tuple(
                edge_id
                for index, edge_id in enumerate(hyperedge_axis)
                if state.state_index & (1 << index)
            )
            if state.open_hyperedge_ids != expected:
                raise _validation_error(
                    "hypergraph state IDs do not match their state index"
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(num=int(value.p), den=int(value.q))


def _connected_in_incidence_graph(
    *,
    open_edges: tuple[tuple[str, ...], ...],
    terminals: tuple[str, str],
) -> bool:
    """Chain connectivity through intersecting open hyperedges."""

    reached = {terminals[0]}
    changed = True
    while changed:
        changed = False
        for members in open_edges:
            member_set = set(members)
            if member_set & reached and not member_set <= reached:
                reached |= member_set
                changed = True
    return terminals[1] in reached


def _admit_hypergraph_request(
    request: HypergraphBondReliabilitySource,
) -> None:
    if len(request.hypergraph.edges) > MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES:
        raise OperationResourceAdmissionError(
            location=("hypergraph", "edges"),
            code="probability.hypergraph_reliability.hyperedge_bound",
            message=(
                "hypergraph reliability exceeds the "
                f"{MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES}-hyperedge bound"
            ),
        )
    if tuple(item.hyperedge_id for item in request.hyperedge_probabilities) != tuple(
        edge_id for edge_id, _ in request.hypergraph.edges
    ):
        raise OperationDomainValidationError(
            location=("hyperedge_probabilities",),
            code="probability.hypergraph_reliability.probability_binding",
            message=(
                "hyperedge probabilities must cover hyperedges in canonical order"
            ),
        )
    if any(
        not 0 <= item.open_probability.as_fraction() <= 1
        for item in request.hyperedge_probabilities
    ):
        raise OperationDomainValidationError(
            location=("hyperedge_probabilities",),
            code="probability.hypergraph_reliability.probability_range",
            message="hypergraph reliability probabilities must lie in [0, 1]",
        )

    # Admit arithmetic height and the complete state ledger together.  The
    # complement branch has numerator den-num, so this maximum covers both
    # open and closed factors for every state.  The powerset sum contributes
    # at most decimal-digit-length(state_count) additional digits.
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
        for item in request.hyperedge_probabilities
    )
    state_count = 1 << len(request.hypergraph.edges)
    incidence_count = sum(len(members) for _, members in request.hypergraph.edges)
    relevant_vertices = {
        *request.terminals,
        *(vertex for _, members in request.hypergraph.edges for vertex in members),
    }
    logical_work = state_count * (
        4 * len(request.hypergraph.edges)
        + 4 * len(relevant_vertices)
        + 4 * incidence_count
        + 16
    )
    if logical_work > MAX_HYPERGRAPH_RELIABILITY_LOGICAL_WORK:
        raise OperationResourceAdmissionError(
            location=("states",),
            code="probability.hypergraph_reliability.work_bound",
            message="hypergraph reliability enumeration exceeds its work bound",
        )
    rational_digits = factor_digits + len(str(state_count))
    if rational_digits > MAX_HYPERGRAPH_RELIABILITY_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("hyperedge_probabilities",),
            code="probability.hypergraph_reliability.rational_height_bound",
            message=(
                "hypergraph reliability state masses exceed the admitted exact "
                f"{MAX_HYPERGRAPH_RELIABILITY_RATIONAL_DIGITS}-digit result bound"
            ),
        )

    hyperedge_count = len(request.hypergraph.edges)
    state_memberships = hyperedge_count * (state_count // 2) if hyperedge_count else 0
    label_units = sum(len(vertex) for vertex in request.hypergraph.vertices) + sum(
        len(edge_id) for edge_id, _ in request.hypergraph.edges
    )
    source_units = (
        label_units
        + sum(len(members) for _, members in request.hypergraph.edges)
        + 4 * len(request.hyperedge_probabilities)
        + 2
    )
    ledger_units = (
        source_units
        + state_count * 4
        + state_memberships
        + 2 * (state_count + 1) * rational_digits
    )
    if ledger_units > MAX_HYPERGRAPH_RELIABILITY_LEDGER_UNITS:
        raise OperationResourceAdmissionError(
            location=("states",),
            code="probability.hypergraph_reliability.output_bound",
            message=(
                "complete hypergraph-reliability ledger exceeds its retained "
                "allocation bound"
            ),
        )


def compute_hypergraph_bond_connection_probability(
    request: HypergraphBondReliabilitySource,
) -> HypergraphBondConnectionProbabilityResult:
    """Compute exact terminal connectivity over every open hyperedge subset."""

    from flint import fmpq

    try:
        source = HypergraphBondReliabilitySource.model_validate(request.model_dump())
    except ValidationError as exc:
        error = exc.errors()[0]
        raise OperationDomainValidationError(
            location=tuple(error["loc"]),
            code=str(error["type"]),
            message=str(error["msg"]),
        ) from None
    _admit_hypergraph_request(source)
    edges = tuple(members for _, members in source.hypergraph.edges)
    probabilities = tuple(
        fmpq(
            item.open_probability.as_fraction().numerator,
            item.open_probability.as_fraction().denominator,
        )
        for item in source.hyperedge_probabilities
    )
    states: list[HypergraphBondReliabilityState] = []
    connection_probability = fmpq(0)
    for state_index in range(1 << len(edges)):
        open_edges = tuple(
            members for index, members in enumerate(edges) if state_index & (1 << index)
        )
        state_probability = fmpq(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        connected = _connected_in_incidence_graph(
            open_edges=open_edges, terminals=source.terminals
        )
        if connected:
            connection_probability += state_probability
        states.append(
            HypergraphBondReliabilityState(
                state_index=state_index,
                open_hyperedge_ids=tuple(
                    source.hypergraph.edges[index][0]
                    for index in range(len(edges))
                    if state_index & (1 << index)
                ),
                terminals_connected=connected,
                state_probability=_wire(state_probability),
            )
        )
    return HypergraphBondConnectionProbabilityResult._from_kernel(
        source=source,
        connection_probability=_wire(connection_probability),
        hyperedge_count=len(edges),
        visited_states=len(states),
        states=tuple(states),
    )


HYPERGRAPH_BOND_CONNECTION_PROBABILITY_OPERATION = MathTool(
    operation_id="probability.hypergraph_bond_reliability.connection_probability.compute",
    title="Compute exact terminal reliability of a finite hypergraph",
    description=(
        "Given a bounded finite simple hypergraph, one exact independent open "
        "probability for every hyperedge, and two distinct terminals, enumerate "
        "every hyperedge subset and return the exact probability that the two "
        "terminals connect through a chain of open hyperedges whose successive "
        "members intersect, together with the complete canonical state ledger."
    ),
    request_type=HypergraphBondReliabilitySource,
    result_type=HypergraphBondConnectionProbabilityResult,
    run=compute_hypergraph_bond_connection_probability,
    tags=("probability", "reliability", "hypergraph", "exact"),
    examples=(
        OperationExample(
            name="two_hyperedges_through_a_bridge",
            description=(
                "Terminals a and d connect when both hyperedges ab and cd are "
                "open and share vertex b=c, so the probability is 1/4."
            ),
            input={
                "hypergraph": {
                    "vertices": ["a", "b", "d"],
                    "edges": [["ab", ["a", "b"]], ["bd", ["b", "d"]]],
                },
                "hyperedge_probabilities": [
                    {
                        "hyperedge_id": "ab",
                        "open_probability": {"num": "1", "den": "2"},
                    },
                    {
                        "hyperedge_id": "bd",
                        "open_probability": {"num": "1", "den": "2"},
                    },
                ],
                "terminals": ["a", "d"],
            },
        ),
    ),
)


__all__ = [
    "HYPERGRAPH_BOND_CONNECTION_PROBABILITY_OPERATION",
    "MAX_HYPERGRAPH_RELIABILITY_HYPEREDGES",
    "MAX_HYPERGRAPH_RELIABILITY_LEDGER_UNITS",
    "MAX_HYPERGRAPH_RELIABILITY_LOGICAL_WORK",
    "MAX_HYPERGRAPH_RELIABILITY_STATES",
    "HyperedgeOpenProbability",
    "HypergraphBondConnectionProbabilityResult",
    "HypergraphBondReliabilitySource",
    "HypergraphBondReliabilityState",
    "compute_hypergraph_bond_connection_probability",
]
