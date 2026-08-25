"""Exact directed bond-reliability operation contract and kernel."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.directed._models import DirectedGraph
from jacobian.math.probability._models import MAX_INPUT_RATIONAL_DIGITS


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("probability.model_invariant", message)


MAX_DIRECTED_BOND_RELIABILITY_ARCS = 12
MAX_DIRECTED_BOND_RELIABILITY_STATES = 1 << MAX_DIRECTED_BOND_RELIABILITY_ARCS
MAX_DIRECTED_BOND_RELIABILITY_LEDGER_BYTES = 9 * 1024 * 1024
# One state mass has at most one numerator and denominator factor per arc.
# Summing at most 2**arcs masses can add at most ``arcs`` decimal digits.
MAX_DIRECTED_BOND_RELIABILITY_RATIONAL_DIGITS = (
    MAX_INPUT_RATIONAL_DIGITS * MAX_DIRECTED_BOND_RELIABILITY_ARCS
    + MAX_DIRECTED_BOND_RELIABILITY_ARCS
)
# The producer enumerates every arc subset and result validation replays it.
# Per state and pass it selects open arcs, evaluates every probability,
# collects the endpoints of open arcs into the relevant vertex set (two
# endpoint visits per open arc), adds those arcs to the directed graph, and
# traverses them: six arc visits. Traversal materializes only the relevant
# vertices -- the two terminals plus both endpoints of every open arc, at most
# ``2 * arcs + 2`` because a state's open arcs are a subset of the declared
# arcs -- inserting each once and visiting each once during the search;
# four relevant-vertex visits charge that with margin.
MAX_DIRECTED_BOND_RELIABILITY_RELEVANT_VERTICES = (
    2 * MAX_DIRECTED_BOND_RELIABILITY_ARCS + 2
)
MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK = (
    2
    * MAX_DIRECTED_BOND_RELIABILITY_STATES
    * (
        7 * MAX_DIRECTED_BOND_RELIABILITY_ARCS
        + 4 * MAX_DIRECTED_BOND_RELIABILITY_RELEVANT_VERTICES
        + 3
    )
    + 8
)
# Declared vertex labels are transport scalars; they do not expand the
# directed-state traversal beyond the relevant vertices charged above.
MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES = (1 << 53) - 1


class DirectedBondReliabilityArcProbability(StrictModel):
    """The independent open probability attached to one directed arc."""

    arc: tuple[StrictInt, StrictInt]
    open_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_directed_arc_probability(self) -> Self:
        if self.arc[0] == self.arc[1]:
            raise _validation_error("directed reliability arcs must not be self-loops")
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_INPUT_RATIONAL_DIGITS,
            label="directed bond reliability arc probability",
        )
        if not 0 <= self.open_probability.as_fraction() <= 1:
            raise _validation_error(
                "directed bond reliability probabilities must lie in [0, 1]"
            )
        return self


def _directed_bond_reliability_graph_schema() -> JsonSchemaValue:
    """Project the bond-reliability envelope onto the shared carrier schema."""

    schema = DirectedGraph.model_json_schema()
    schema["description"] = (
        "A structurally valid finite simple directed graph accepted by this "
        f"complete-enumeration operation: at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} "
        f"arcs, and declared vertex labels inside the interoperable JSON "
        f"integer range (at most {MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES}). "
        "Traversal work scales with relevant vertices (terminals plus arc "
        "endpoints), not with declared vertices."
    )
    schema["properties"]["vertex_count"].update(
        maximum=MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES,
    )
    schema["properties"]["edges"].update(maxItems=MAX_DIRECTED_BOND_RELIABILITY_ARCS)
    return schema


DirectedBondReliabilityGraph = Annotated[
    DirectedGraph,
    WithJsonSchema(_directed_bond_reliability_graph_schema()),
]


class DirectedBondConnectionProbabilitySource(StrictModel):
    __doc__ = f"""One finite directed bond-percolation source in canonical arc order.

    Sources have at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} arcs; traversal and replay materialize
    only relevant vertices (the terminals plus arc endpoints), so sparse
    sources may declare very large vertex counts within the published
    declared-vertex label bound. The probability map must contain every
    graph arc exactly once and is empty for an edgeless source, and
    source/target are distinct declared vertices. Input arc rows are
    normalized to lexicographic arc order before state indices and result
    records are assigned.
    """

    graph: DirectedBondReliabilityGraph = Field(
        description=(
            f"A directed graph with at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} arcs for this "
            "complete-enumeration operation. Traversal work scales with "
            "relevant vertices (terminals plus arc endpoints), not with "
            "declared vertices."
        )
    )
    arc_probabilities: tuple[DirectedBondReliabilityArcProbability, ...] = Field(
        max_length=MAX_DIRECTED_BOND_RELIABILITY_ARCS,
        description=(
            "One independent open probability for every graph arc exactly once. "
            "Input rows are accepted in any order and normalized to lexicographic "
            "arc order."
        ),
    )
    source: StrictInt = Field(
        description="A declared source vertex, distinct from target."
    )
    target: StrictInt = Field(
        description="A declared target vertex, distinct from source."
    )

    @model_validator(mode="after")
    def require_bounded_fully_weighted_directed_graph(self) -> Self:
        if len(self.graph.edges) > MAX_DIRECTED_BOND_RELIABILITY_ARCS:
            raise _validation_error(
                "directed bond reliability exceeds the "
                f"{MAX_DIRECTED_BOND_RELIABILITY_ARCS}-arc bound"
            )
        if self.source == self.target or any(
            vertex < 0 or vertex >= self.graph.vertex_count
            for vertex in (self.source, self.target)
        ):
            raise _validation_error(
                "source and target must be distinct declared graph vertices"
            )

        probabilities_by_arc = {
            item.arc: item.open_probability for item in self.arc_probabilities
        }
        if len(probabilities_by_arc) != len(self.arc_probabilities) or frozenset(
            probabilities_by_arc
        ) != frozenset(self.graph.edges):
            raise _validation_error(
                "arc probabilities must contain every directed graph arc exactly once"
            )

        canonical_arcs = tuple(sorted(self.graph.edges))
        object.__setattr__(
            self,
            "graph",
            DirectedGraph(
                vertex_count=self.graph.vertex_count,
                edges=canonical_arcs,
            ),
        )
        object.__setattr__(
            self,
            "arc_probabilities",
            tuple(
                DirectedBondReliabilityArcProbability(
                    arc=arc,
                    open_probability=probabilities_by_arc[arc],
                )
                for arc in canonical_arcs
            ),
        )

        arc_count = len(canonical_arcs)
        state_count = 1 << arc_count
        relevant_vertices = len(
            {self.source, self.target}
            | {vertex for arc in canonical_arcs for vertex in arc}
        )
        logical_work = 2 * state_count * (7 * arc_count + 4 * relevant_vertices + 3) + 8
        if logical_work > MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK:
            raise _validation_error(
                "directed bond reliability exceeds the complete producer and "
                "replay work budget"
            )
        if self.graph.vertex_count > MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES:
            raise _validation_error(
                "declared vertex labels exceed the interoperable JSON integer "
                f"range of {MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES}"
            )
        repeated_arc_bytes = (state_count // 2) * sum(
            len(encode_strict_json(list(arc))) + 1 for arc in canonical_arcs
        )
        # Every state mass multiplies one open or one closed factor per arc,
        # so a state's numerator and denominator digit counts are bounded by
        # the corresponding sums over its selected factors, and each factor's
        # digits occur in exactly half of the powerset states.
        per_arc_numerator_bytes = sum(
            len(format_canonical_integer(item.open_probability.as_fraction().numerator))
            + len(
                format_canonical_integer(
                    (1 - item.open_probability.as_fraction()).numerator
                )
            )
            for item in self.arc_probabilities
        )
        per_arc_denominator_bytes = sum(
            len(
                format_canonical_integer(
                    item.open_probability.as_fraction().denominator
                )
            )
            + len(
                format_canonical_integer(
                    (1 - item.open_probability.as_fraction()).denominator
                )
            )
            for item in self.arc_probabilities
        )
        maximum_state_template = {
            "state_index": 0,
            "open_arcs": [],
            "source_reaches_target": False,
            "state_probability": {"num": "", "den": ""},
        }
        fixed_state_bytes = len(encode_strict_json(maximum_state_template))
        source_bytes = len(
            encode_strict_json(
                {
                    "graph": {
                        "vertex_count": self.graph.vertex_count,
                        "edges": [list(arc) for arc in canonical_arcs],
                    },
                    "arc_probabilities": [
                        {
                            "arc": list(item.arc),
                            "open_probability": item.open_probability.model_dump(),
                        }
                        for item in self.arc_probabilities
                    ],
                    "source": self.source,
                    "target": self.target,
                }
            )
        )
        estimated_ledger_bytes = (
            source_bytes
            + repeated_arc_bytes
            # Each record adds its index digits and at least one numerator
            # and denominator character beyond the empty template, plus one
            # ledger separator byte.
            + state_count * (fixed_state_bytes + len(str(state_count - 1)) + 2 + 1)
            + (state_count // 2) * (per_arc_numerator_bytes + per_arc_denominator_bytes)
            + 16 * 1024
        )
        if estimated_ledger_bytes > MAX_DIRECTED_BOND_RELIABILITY_LEDGER_BYTES:
            raise _validation_error(
                "directed bond reliability request can exceed the complete ledger "
                f"budget of {MAX_DIRECTED_BOND_RELIABILITY_LEDGER_BYTES} bytes"
            )
        return self


class DirectedBondConnectionProbabilityRequest(StrictModel):
    __doc__ = f"""Compute directed source-to-target bond connection probability.

    The request admits at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} arcs, requires one probability for every
    arc exactly once (empty for an edgeless graph), and requires distinct
    declared source and target vertices. Traversal and replay materialize
    only relevant vertices, so sparse graphs may declare very large vertex
    counts within the published declared-vertex label bound. It normalizes
    arc rows lexicographically, so state indices and the source-bound result
    do not depend on input order.
    """

    graph: DirectedBondReliabilityGraph = Field(
        description=(
            f"A directed graph with at most {MAX_DIRECTED_BOND_RELIABILITY_ARCS} arcs for this "
            "complete-enumeration operation. Traversal work scales with "
            "relevant vertices (terminals plus arc endpoints), not with "
            "declared vertices."
        )
    )
    arc_probabilities: tuple[DirectedBondReliabilityArcProbability, ...] = Field(
        max_length=MAX_DIRECTED_BOND_RELIABILITY_ARCS,
        description=(
            "One independent open probability for every graph arc exactly once. "
            "Input rows are accepted in any order and normalized to lexicographic "
            "arc order."
        ),
    )
    source: StrictInt = Field(
        description="A declared source vertex, distinct from target."
    )
    target: StrictInt = Field(
        description="A declared target vertex, distinct from source."
    )
    event: Literal["DIRECTED_PATH_EXISTS"] = "DIRECTED_PATH_EXISTS"

    @model_validator(mode="after")
    def require_canonical_source(self) -> Self:
        canonical_source = DirectedBondConnectionProbabilitySource(
            graph=self.graph,
            arc_probabilities=self.arc_probabilities,
            source=self.source,
            target=self.target,
        )
        object.__setattr__(self, "graph", canonical_source.graph)
        object.__setattr__(
            self, "arc_probabilities", canonical_source.arc_probabilities
        )
        return self


class DirectedBondReliabilityState(StrictModel):
    """One exact directed arc-subset state from a bond-percolation source."""

    state_index: StrictInt = Field(
        ge=0,
        lt=MAX_DIRECTED_BOND_RELIABILITY_STATES,
    )
    open_arcs: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        max_length=MAX_DIRECTED_BOND_RELIABILITY_ARCS
    )
    source_reaches_target: bool
    state_probability: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_probability(self) -> Self:
        require_bounded_rational(
            self.state_probability,
            max_digits=MAX_DIRECTED_BOND_RELIABILITY_RATIONAL_DIGITS,
            label="directed bond reliability state probability",
        )
        if not 0 <= self.state_probability.as_fraction() <= 1:
            raise _validation_error(
                "directed bond reliability state probability must lie in [0, 1]"
            )
        return self


class DirectedBondConnectionProbabilityResult(StrictModel):
    """An exact, complete, source-bound directed bond reliability result."""

    source: DirectedBondConnectionProbabilitySource
    connection_probability: CanonicalRational
    arc_count: StrictInt = Field(ge=0, le=MAX_DIRECTED_BOND_RELIABILITY_ARCS)
    visited_states: StrictInt = Field(
        ge=1,
        le=MAX_DIRECTED_BOND_RELIABILITY_STATES,
    )
    states: tuple[DirectedBondReliabilityState, ...] = Field(
        min_length=1,
        max_length=MAX_DIRECTED_BOND_RELIABILITY_STATES,
    )
    event: Literal["DIRECTED_PATH_EXISTS"] = "DIRECTED_PATH_EXISTS"
    arc_independence: Literal["INDEPENDENT_BERNOULLI"] = "INDEPENDENT_BERNOULLI"
    enumeration: Literal["COMPLETE_ARC_SUBSETS"] = "COMPLETE_ARC_SUBSETS"
    completeness: Literal["COMPLETE"] = "COMPLETE"
    truncated: Literal[False] = False
    termination_reason: Literal["EXHAUSTED"] = "EXHAUSTED"
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"

    @model_validator(mode="after")
    def bind_to_directed_bond_source(self) -> Self:
        require_bounded_rational(
            self.connection_probability,
            max_digits=MAX_DIRECTED_BOND_RELIABILITY_RATIONAL_DIGITS,
            label="directed bond connection probability",
        )
        connection_probability, expected_states = (
            _directed_bond_connection_probability_data(self.source)
        )
        if self.arc_count != len(self.source.graph.edges):
            raise _validation_error("arc_count must match the source graph")
        if self.visited_states != 1 << self.arc_count:
            raise _validation_error("visited state count is not the full arc powerset")
        if len(self.states) != self.visited_states:
            raise _validation_error("state ledger length does not match visited states")
        if tuple(item.state_index for item in self.states) != tuple(
            range(self.visited_states)
        ):
            raise _validation_error(
                "state ledger indices must be complete and canonical"
            )
        if len(expected_states) != len(self.states):
            raise _validation_error("state ledger length does not match source replay")
        for state, expected in zip(self.states, expected_states, strict=True):
            open_arcs, reaches_target, state_probability = expected
            if state.open_arcs != open_arcs:
                raise _validation_error("state open arcs do not match source subset")
            if state.source_reaches_target != reaches_target:
                raise _validation_error(
                    "state reachability does not match source subset"
                )
            if state.state_probability.as_fraction() != state_probability:
                raise _validation_error(
                    "state probability does not match source subset"
                )
        if self.connection_probability.as_fraction() != connection_probability:
            raise _validation_error(
                "connection probability does not match source replay"
            )
        return self


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.p)),
        den=format_canonical_integer(int(value.q)),
    )


def _directed_bond_connection_probability_data(
    source: DirectedBondConnectionProbabilitySource,
) -> tuple[Fraction, tuple[tuple[tuple[tuple[int, int], ...], bool, Fraction], ...]]:
    """Replay the full directed bond-percolation source with exact rationals.

    Directed reachability is delegated to the directed-graph owner.  The
    standard-library ``Fraction`` replay is deliberately independent from the
    Python-FLINT producer used by the public operation below.
    """

    probabilities = tuple(
        item.open_probability.as_fraction() for item in source.arc_probabilities
    )
    states: list[tuple[tuple[tuple[int, int], ...], bool, Fraction]] = []
    connection_probability = Fraction()
    for state_index in range(1 << len(source.graph.edges)):
        open_arcs = tuple(
            arc
            for index, arc in enumerate(source.graph.edges)
            if state_index & (1 << index)
        )
        state_probability = Fraction(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        reaches_target = _directed_path_exists(
            arcs=open_arcs,
            source=source.source,
            target=source.target,
        )
        if reaches_target:
            connection_probability += state_probability
        states.append((open_arcs, reaches_target, state_probability))
    return connection_probability, tuple(states)


def _directed_path_exists(
    *,
    arcs: tuple[tuple[int, int], ...],
    source: int,
    target: int,
) -> bool:
    """Test one admitted state without another operation's carrier envelope."""

    import networkx as nx

    relevant_vertices = {source, target}
    relevant_vertices.update(vertex for arc in arcs for vertex in arc)

    graph: Any = nx.DiGraph()
    graph.add_nodes_from(relevant_vertices)
    graph.add_edges_from(arcs)
    return nx.has_path(graph, source, target)


def _directed_bond_connection_probability(
    request: DirectedBondConnectionProbabilityRequest,
) -> DirectedBondConnectionProbabilityResult:
    """Compute exact directed source-to-target bond reliability with FLINT."""

    from flint import fmpq

    source = DirectedBondConnectionProbabilitySource(
        graph=request.graph,
        arc_probabilities=request.arc_probabilities,
        source=request.source,
        target=request.target,
    )
    probabilities = tuple(
        fmpq(
            item.open_probability.as_fraction().numerator,
            item.open_probability.as_fraction().denominator,
        )
        for item in source.arc_probabilities
    )
    states: list[DirectedBondReliabilityState] = []
    connection_probability = fmpq(0)
    for state_index in range(1 << len(source.graph.edges)):
        open_arcs = tuple(
            arc
            for index, arc in enumerate(source.graph.edges)
            if state_index & (1 << index)
        )
        state_probability = fmpq(1)
        for index, probability in enumerate(probabilities):
            state_probability *= (
                probability if state_index & (1 << index) else 1 - probability
            )
        reaches_target = _directed_path_exists(
            arcs=open_arcs,
            source=source.source,
            target=source.target,
        )
        if reaches_target:
            connection_probability += state_probability
        states.append(
            DirectedBondReliabilityState(
                state_index=state_index,
                open_arcs=open_arcs,
                source_reaches_target=reaches_target,
                state_probability=_wire(state_probability),
            )
        )
    return DirectedBondConnectionProbabilityResult(
        source=source,
        connection_probability=_wire(connection_probability),
        arc_count=len(source.graph.edges),
        visited_states=len(states),
        states=tuple(states),
    )


DIRECTED_BOND_CONNECTION_PROBABILITY_OPERATION = MathTool(
    operation_id="probability.digraph_bond_reliability.connection_probability.compute",
    title="Exact finite directed bond connection probability",
    description=(
        "Compute the exact probability of a directed path from one stated "
        "source vertex to one stated target vertex in a bounded directed "
        "graph with independent rational arc-open probabilities. The complete "
        "arc-subset ledger is source-bound and replayed."
    ),
    request_type=DirectedBondConnectionProbabilityRequest,
    result_type=DirectedBondConnectionProbabilityResult,
    run=_directed_bond_connection_probability,
    tags=(
        "probability",
        "directed-graph",
        "reliability",
        "bond-percolation",
        "connection",
        "reachability",
        "exact",
        "bounded",
        "networkx",
        "python-flint",
    ),
    examples=(
        example(
            "two_arc_directed_series",
            "Compute the exact probability of the directed path 0 -> 1 -> 2; "
            "each directed graph arc has one independent rational open probability.",
            {
                "graph": {
                    "vertex_count": 3,
                    "edges": [[0, 1], [1, 2]],
                },
                "arc_probabilities": [
                    {
                        "arc": [0, 1],
                        "open_probability": {"num": "2", "den": "3"},
                    },
                    {
                        "arc": [1, 2],
                        "open_probability": {"num": "3", "den": "5"},
                    },
                ],
                "source": 0,
                "target": 2,
            },
        ),
    ),
)


__all__ = [
    "DIRECTED_BOND_CONNECTION_PROBABILITY_OPERATION",
    "MAX_DIRECTED_BOND_RELIABILITY_ARCS",
    "MAX_DIRECTED_BOND_RELIABILITY_DECLARED_VERTICES",
    "MAX_DIRECTED_BOND_RELIABILITY_LOGICAL_WORK",
    "MAX_DIRECTED_BOND_RELIABILITY_RELEVANT_VERTICES",
    "MAX_DIRECTED_BOND_RELIABILITY_STATES",
    "DirectedBondConnectionProbabilityRequest",
    "DirectedBondConnectionProbabilityResult",
    "DirectedBondConnectionProbabilitySource",
    "DirectedBondReliabilityArcProbability",
    "DirectedBondReliabilityState",
]
