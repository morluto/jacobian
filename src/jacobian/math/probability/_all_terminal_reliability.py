"""Wire contract for exact all-terminal graph reliability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability.all_terminal_reliability import (
    MAX_ALL_TERMINAL_RELIABILITY_EDGES,
    MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS,
    MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS,
    MAX_ALL_TERMINAL_RELIABILITY_STATES,
    _compute_all_terminal_reliability,
    _require_bounded_problem,
    _require_source_bound_result,
)


class AllTerminalReliabilityRequest(StrictModel):
    """One nonempty bounded graph and one uniform exact edge-up probability."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute all-terminal reliability for one nonempty canonical "
                "simple undirected graph. Every edge is independently open with "
                "the same exact rational `open_probability`. The graph may have "
                "at most 20 edges, bounding each complete enumeration to 2^20 "
                "states. The retained graph plus the fixed exact-result envelope "
                "must fit the canonical output limit."
            )
        }
    )

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Nonempty canonical simple undirected graph with at most 20 edges. "
            "Isolated declared vertices participate in the all-terminal event. "
            "The retained graph plus fixed result headroom must fit the canonical "
            "output limit."
        )
    )
    open_probability: CanonicalRational = Field(
        description=(
            "Uniform independent probability that each graph edge is open, as "
            "an exact canonical rational in [0, 1] with at most 128 digits per "
            "component."
        )
    )
    event: Literal["ALL_VERTICES_CONNECTED"] = "ALL_VERTICES_CONNECTED"

    @model_validator(mode="after")
    def require_bounded_complete_enumeration(self) -> Self:
        all_terminal_reliability_input = self.open_probability.as_fraction()
        # The native boundary owns all mathematical and two-pass work admission.
        _require_bounded_problem(self.graph, all_terminal_reliability_input)
        return self


class AllTerminalReliabilityWireResult(StrictModel):
    """Source-bound exact probability with its connected-subgraph profile."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Exact all-terminal reliability bound to the retained graph and "
                "uniform edge probability. Entry k of "
                "`connected_spanning_subgraph_counts` is the number of connected "
                "spanning edge subsets containing exactly k edges; result "
                "validation replays the complete bounded enumeration."
            )
        }
    )

    graph: SimpleUndirectedGraph
    open_probability: CanonicalRational
    connected_spanning_subgraph_counts: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_ALL_TERMINAL_RELIABILITY_EDGES + 1,
        description=(
            "Counts indexed by open-edge cardinality k=0..m. They reconstruct "
            "R_G(p)=sum_k c_k p^k (1-p)^(m-k)."
        ),
    )
    reliability_probability: CanonicalRational
    visited_states: StrictInt = Field(
        ge=1,
        le=MAX_ALL_TERMINAL_RELIABILITY_STATES,
        description="The number of edge-subset states exhaustively visited, 2^m.",
    )
    event: Literal["ALL_VERTICES_CONNECTED"] = "ALL_VERTICES_CONNECTED"

    @model_validator(mode="before")
    @classmethod
    def bound_raw_coefficients(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw_counts = value.get("connected_spanning_subgraph_counts")
        if isinstance(raw_counts, (list, tuple)):
            if len(raw_counts) > MAX_ALL_TERMINAL_RELIABILITY_EDGES + 1:
                raise ValueError(
                    "connected-spanning-subgraph profile exceeds the edge bound"
                )
            max_digits = len(str(MAX_ALL_TERMINAL_RELIABILITY_STATES))
            if any(
                isinstance(item, str) and len(item.lstrip("-")) > max_digits
                for item in raw_counts
            ):
                raise ValueError(
                    "connected-spanning-subgraph count exceeds the state bound"
                )
        return value

    @model_validator(mode="after")
    def bind_to_source_graph(self) -> Self:
        probability = self.open_probability.as_fraction()
        require_bounded_rational(
            self.open_probability,
            max_digits=MAX_ALL_TERMINAL_RELIABILITY_INPUT_DIGITS,
            label="all-terminal reliability open probability",
        )
        require_bounded_rational(
            self.reliability_probability,
            max_digits=MAX_ALL_TERMINAL_RELIABILITY_RESULT_DIGITS,
            label="all-terminal reliability result probability",
        )
        actual_counts = tuple(
            parse_canonical_integer(value)
            for value in self.connected_spanning_subgraph_counts
        )
        _require_source_bound_result(
            self.graph,
            probability,
            actual_counts,
            self.reliability_probability.as_fraction(),
            self.visited_states,
        )
        return self


def compute_all_terminal_reliability(
    request: AllTerminalReliabilityRequest,
) -> AllTerminalReliabilityWireResult:
    probability = request.open_probability.as_fraction()
    counts, reliability_probability, visited_states = _compute_all_terminal_reliability(
        request.graph, probability
    )
    return AllTerminalReliabilityWireResult(
        graph=request.graph,
        open_probability=request.open_probability,
        connected_spanning_subgraph_counts=tuple(
            format_canonical_integer(count) for count in counts
        ),
        reliability_probability=CanonicalRational.from_fraction(
            reliability_probability
        ),
        visited_states=visited_states,
    )


ALL_TERMINAL_RELIABILITY_OPERATION = MathTool(
    operation_id="probability.graph_reliability.all_terminal.compute",
    version="1",
    title="Exact bounded all-terminal graph reliability",
    description=(
        "Compute the exact probability that the spanning subgraph on every "
        "declared vertex is connected when each edge is independently open with "
        "one uniform rational probability. Return the complete connected-spanning-"
        "subgraph count vector as a source-bound reconstruction value."
    ),
    request_type=AllTerminalReliabilityRequest,
    result_type=AllTerminalReliabilityWireResult,
    run=compute_all_terminal_reliability,
    tags=(
        "probability",
        "graph",
        "reliability",
        "all-terminal",
        "connected-spanning-subgraph",
        "exact",
        "bounded",
    ),
    examples=(
        example(
            "fair_edge_triangle_all_terminal_reliability",
            (
                "Compute the exact probability that a fair-edge triangle is "
                "connected on all declared vertices; the graph must be nonempty, "
                "have at most 20 edges, use one rational edge probability, and "
                "fit the retained-result output limit."
            ),
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                },
                "open_probability": {"num": "1", "den": "2"},
            },
        ),
    ),
)


__all__ = ["ALL_TERMINAL_RELIABILITY_OPERATION"]
