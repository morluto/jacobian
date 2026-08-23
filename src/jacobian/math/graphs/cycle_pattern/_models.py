"""Typed wire contracts for graph cycle and subgraph-pattern operations.

Graphs are carried as the domain-owned canonical ``SimpleUndirectedGraph``
value so serialized producer output composes into these requests unchanged;
integer indexing stays private to the bounded search kernel.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, ValidationInfo, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_CYCLE_GRAPH_ORDER = 64

# Serialized canonical results stay under Jacobian's 10 MiB transport
# ceiling with room for the OperationResult envelope around them; a
# schema-valid request must never produce an output the host cannot
# canonicalize.
MAX_CYCLE_PATTERN_RESULT_BYTES = 9 * 1024 * 1024
_RESULT_ENVELOPE_SLACK_BYTES = 1_024

# Validation-context key carrying one executed search's budget. The
# operation attaches the very budget whose completion decided a negative
# result, so validation reuses that first-pass evidence instead of paying
# a second exhaustive pass per request. Payloads decoded without this
# programmatic context always replay the full bounded search.
_FIRST_PASS_BUDGET_CONTEXT_KEY = "cycle_pattern_first_pass_budget"


def _require_admissible_order(graph: SimpleUndirectedGraph) -> None:
    if len(graph.vertices) > MAX_CYCLE_GRAPH_ORDER:
        raise ValueError(
            "graphs are bounded to at most "
            f"{MAX_CYCLE_GRAPH_ORDER} vertices so the exhaustive searches "
            "keep a declared work and result bound"
        )


def _encoded_graph_bytes(graph: SimpleUndirectedGraph) -> int:
    """Serialized size of one retained graph value."""
    return len(canonicalize_json(graph.model_dump(mode="json")))


def _encoded_label_bytes(label: str) -> int:
    """Serialized size of one label as a JSON string value."""
    return len(encode_strict_json(label))


def _label_wire_costs_desc(graph: SimpleUndirectedGraph) -> list[int]:
    """Exact canonical wire costs of every vertex label, most expensive first.

    Canonical JSON escapes control characters to six wire bytes, so raw
    UTF-8 length can undercharge labels; each cost is measured on the
    encoded form the result will actually carry.
    """
    return sorted(
        (_encoded_label_bytes(vertex) for vertex in graph.vertices),
        reverse=True,
    )


def _require_admissible_cycle_result_bytes(
    graph: SimpleUndirectedGraph, length: int
) -> None:
    """Reserve transport space for the complete cycle result.

    The result retains the source graph and a positive witness repeats up
    to ``length`` vertex labels, so admission predicts the canonical
    encoding of everything the result carries and rejects requests that
    cannot fit the output budget. Witness repetitions are charged at the
    exact encoded costs of the most expensive labels.
    """
    witness_bytes = sum(_label_wire_costs_desc(graph)[:length])
    predicted = (
        _encoded_graph_bytes(graph) + witness_bytes + _RESULT_ENVELOPE_SLACK_BYTES
    )
    if predicted > MAX_CYCLE_PATTERN_RESULT_BYTES:
        raise ValueError(
            "cycle request result would exceed the "
            f"{MAX_CYCLE_PATTERN_RESULT_BYTES}-byte canonical result budget"
        )


def _require_admissible_pattern_result_bytes(
    host: SimpleUndirectedGraph, pattern: SimpleUndirectedGraph
) -> None:
    """Reserve transport space for the complete subgraph-pattern result.

    The result retains both graphs plus one mapping entry per pattern
    vertex; admission predicts that aggregate encoding against the output
    budget before any search runs. Mapping entries pair each pattern label
    with a distinct host label, so charging the largest encoded costs from
    each side independently bounds every injective mapping.
    """
    pattern_costs = _label_wire_costs_desc(pattern)
    host_costs = _label_wire_costs_desc(host)
    mapping_bytes = (
        sum(pattern_costs)
        + sum(host_costs[: len(pattern_costs)])
        + 5 * len(pattern.vertices)
    )
    predicted = (
        _encoded_graph_bytes(host)
        + _encoded_graph_bytes(pattern)
        + mapping_bytes
        + _RESULT_ENVELOPE_SLACK_BYTES
    )
    if predicted > MAX_CYCLE_PATTERN_RESULT_BYTES:
        raise ValueError(
            "subgraph pattern request result would exceed the "
            f"{MAX_CYCLE_PATTERN_RESULT_BYTES}-byte canonical result budget"
        )


class FixedLengthCycleRequest(StrictModel):
    """Decide whether a graph contains a simple cycle of a given length."""

    graph: SimpleUndirectedGraph
    length: int = Field(ge=3, le=MAX_CYCLE_GRAPH_ORDER)

    @model_validator(mode="after")
    def require_length_within_bounds(self) -> Self:
        _require_admissible_order(self.graph)
        if self.length > len(self.graph.vertices):
            raise ValueError("cycle length cannot exceed vertex count")
        _require_admissible_cycle_result_bytes(self.graph, self.length)
        return self


def _require_decided_witness_pair(exists: bool | None, witness: object) -> None:
    """A decided search states a claim; exactly an affirmative one carries a witness."""
    if exists is None:
        raise ValueError("a decided search states whether the target exists")
    if exists is (witness is None):
        raise ValueError("exactly an existing witness carries one witness")


def _require_cycle_witness_replay(
    graph: SimpleUndirectedGraph, length: int, cycle: tuple[str, ...]
) -> None:
    """A positive cycle witness must replay as edges of its source graph."""
    if len(cycle) != length:
        raise ValueError("witness cycle length must match the declared length")
    if len(set(cycle)) != length:
        raise ValueError("witness cycle vertices must be distinct")
    names = set(graph.vertices)
    if any(vertex not in names for vertex in cycle):
        raise ValueError("witness cycle vertices must lie in the graph")
    edges = set(graph.edges)
    for index, u in enumerate(cycle):
        v = cycle[(index + 1) % len(cycle)]
        edge = (u, v) if u < v else (v, u)
        if edge not in edges:
            raise ValueError("witness cycle must replay as edges of the source graph")


def _require_negative_cycle_replay(graph: SimpleUndirectedGraph, length: int) -> None:
    """Replay the same bounded exhaustive search the operation ran.

    A decided negative result implies the search completed within its
    deterministic budget, so an identical replay decides within budget as
    well; exhaustion here means the payload cannot be re-verified.
    """
    from jacobian.math.graphs.cycle_pattern._operations import (
        _BudgetExceededError,
        _decide_cycle_bounded,
    )

    try:
        contradicts = _decide_cycle_bounded(graph, length)
    except _BudgetExceededError as error:
        raise ValueError(
            "negative cycle decision cannot be re-verified within "
            "the cycle search budget"
        ) from error
    if contradicts:
        raise ValueError(
            "negative cycle decision contradicts exhaustive search: a k-cycle exists"
        )


class SearchOutcome(StrictModel):
    """Shared outcome discriminator for bounded exhaustive searches."""

    outcome: Literal["DECIDED", "SEARCH_BUDGET_EXCEEDED"] = "DECIDED"
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "DECIDED":
            if self.detail is not None:
                raise ValueError("decided searches carry no failure detail")
        elif self.detail is None:
            raise ValueError("an exceeded search budget carries a detail")
        return self


def _first_pass_evidence_attached(info: ValidationInfo | None) -> bool:
    """Whether the executing operation attached its own completed search.

    Presence of the executed budget under ``_FIRST_PASS_BUDGET_CONTEXT_KEY``
    is programmatic evidence that this exact payload was produced by a
    first pass that already exhausted the admitted search, so validation
    reuses it rather than paying a second exhaustive pass. Any decode
    without that context replays the search in full.
    """
    return (
        info is not None
        and info.context is not None
        and _FIRST_PASS_BUDGET_CONTEXT_KEY in info.context
    )


class FixedLengthCycleResult(SearchOutcome):
    """Whether a simple k-cycle exists, with an explicit witness.

    Carries its source graph so the witness replays against the exact edge
    relation instead of trusting vertex bookkeeping. ``exists`` is ``None``
    exactly for the undecided SEARCH_BUDGET_EXCEEDED outcome.
    """

    graph: SimpleUndirectedGraph
    length: int = Field(ge=3, le=MAX_CYCLE_GRAPH_ORDER)
    exists: bool | None = None
    cycle: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def bind_witness(self, info: ValidationInfo) -> Self:
        _require_admissible_order(self.graph)
        if self.length > len(self.graph.vertices):
            raise ValueError("cycle length cannot exceed vertex count")
        if self.outcome != "DECIDED":
            if self.exists is not None or self.cycle is not None:
                raise ValueError(
                    "a budget-exceeded search returns neither claim nor witness"
                )
            return self
        _require_decided_witness_pair(self.exists, self.cycle)
        if self.cycle is not None:
            _require_cycle_witness_replay(self.graph, self.length, self.cycle)
        elif not _first_pass_evidence_attached(info):
            _require_negative_cycle_replay(self.graph, self.length)
        return self


class SubgraphPatternRequest(StrictModel):
    """Find an injective embedding of a pattern graph into a host graph."""

    host: SimpleUndirectedGraph
    pattern: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_pattern_fits(self) -> Self:
        _require_admissible_order(self.host)
        _require_admissible_order(self.pattern)
        if len(self.pattern.vertices) > len(self.host.vertices):
            raise ValueError("pattern vertex count cannot exceed host vertex count")
        _require_admissible_pattern_result_bytes(self.host, self.pattern)
        return self


class SubgraphEmbedding(StrictModel):
    """One injective embedding of a pattern graph into a host graph.

    The empty mapping is the unique embedding of the empty pattern.
    """

    mapping: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def require_valid_mapping(self) -> Self:
        domain = tuple(src for src, _ in self.mapping)
        codomain = tuple(dst for _, dst in self.mapping)
        if domain != tuple(sorted(domain)):
            raise ValueError("embedding domain must be sorted")
        # The mapping must be a function on distinct pattern vertices before
        # dictionary conversion could silently drop duplicate entries.
        if len(set(domain)) != len(domain):
            raise ValueError("embedding domain vertices must be distinct")
        if len(set(codomain)) != len(codomain):
            raise ValueError("embedding codomain must be injective")
        return self


class SubgraphPatternResult(SearchOutcome):
    """Whether a subgraph embedding exists, with an explicit witness.

    Carries both source graphs so the embedding replays injectivity, domain
    coverage, and exact edge preservation against the inputs.
    """

    host_graph: SimpleUndirectedGraph
    pattern_graph: SimpleUndirectedGraph
    exists: bool | None = None
    embedding: SubgraphEmbedding | None = None

    @model_validator(mode="after")
    def bind_witness(self, info: ValidationInfo) -> Self:
        _require_admissible_order(self.host_graph)
        _require_admissible_order(self.pattern_graph)
        if len(self.pattern_graph.vertices) > len(self.host_graph.vertices):
            raise ValueError("pattern vertex count cannot exceed host vertex count")
        if self.outcome != "DECIDED":
            if self.exists is not None or self.embedding is not None:
                raise ValueError(
                    "a budget-exceeded search returns neither claim nor witness"
                )
            return self
        _require_decided_witness_pair(self.exists, self.embedding)
        if self.embedding is not None:
            _require_embedding_witness_replay(
                self.host_graph, self.pattern_graph, self.embedding
            )
        elif not _first_pass_evidence_attached(info):
            _require_negative_embedding_replay(self.host_graph, self.pattern_graph)
        return self


def _require_embedding_witness_replay(
    host_graph: SimpleUndirectedGraph,
    pattern_graph: SimpleUndirectedGraph,
    embedding: SubgraphEmbedding,
) -> None:
    """A positive embedding must replay injectively and preserve edges."""
    mapping = dict(embedding.mapping)
    if set(mapping) != set(pattern_graph.vertices):
        raise ValueError("embedding must cover every pattern vertex")
    host_names = set(host_graph.vertices)
    if any(hv not in host_names for hv in mapping.values()):
        raise ValueError("embedding codomain must lie in the host graph")
    host_edges = set(host_graph.edges)
    for a, b in pattern_graph.edges:
        ha, hb = mapping[a], mapping[b]
        edge = (ha, hb) if ha < hb else (hb, ha)
        if edge not in host_edges:
            raise ValueError("embedding must preserve every pattern edge in the host")


def _require_negative_embedding_replay(
    host_graph: SimpleUndirectedGraph,
    pattern_graph: SimpleUndirectedGraph,
) -> None:
    """Replay the same degree-pruned bounded search the operation ran.

    A decided negative result implies the search completed within its
    deterministic budget, so an identical replay must decide within budget
    as well; exhaustion here means the payload cannot be re-verified.
    """
    from jacobian.math.graphs.cycle_pattern._operations import (
        _BudgetExceededError,
        _decide_embedding_bounded,
    )

    try:
        contradicts = _decide_embedding_bounded(host_graph, pattern_graph)
    except _BudgetExceededError as error:
        raise ValueError(
            "negative subgraph decision cannot be re-verified "
            "within the embedding search budget"
        ) from error
    if contradicts:
        raise ValueError(
            "negative subgraph decision contradicts existence of an embedding"
        )


__all__ = [
    "MAX_CYCLE_GRAPH_ORDER",
    "MAX_CYCLE_PATTERN_RESULT_BYTES",
    "FixedLengthCycleRequest",
    "FixedLengthCycleResult",
    "SubgraphEmbedding",
    "SubgraphPatternRequest",
    "SubgraphPatternResult",
]
