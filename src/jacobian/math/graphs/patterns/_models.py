"""Typed contracts and admission for induced vertex-subset pattern counts."""

from __future__ import annotations

from math import comb
from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

# One accepted request performs the exact count once.  Independently supplied
# claims may be checked separately by the owner-local verifier, rather than
# causing result construction to replay NetworkX.
COUNT_PASSES = 1

# The subset count separately bounds explicit candidate construction. The
# total work budget below additionally charges exact isomorphism search. These
# are conservative limits for the current pure-Python NetworkX envelope, not
# restrictions on the mathematical definition; widening requires sharper
# backend-specific search bounds and representative boundary measurements.
MAX_INDUCED_PATTERN_SUBSETS_PER_PASS = 5_000
MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS = 64_000_000

# Every returned count is at most the admitted number of candidate subsets.
MAX_INDUCED_PATTERN_COUNT_DIGITS = len(
    format_canonical_integer(MAX_INDUCED_PATTERN_SUBSETS_PER_PASS)
)


def _candidate_subset_count(host_order: int, pattern_order: int) -> int:
    return 0 if pattern_order > host_order else comb(host_order, pattern_order)


def _partial_injection_state_bound(order: int) -> int:
    """Bound every partial bijection state an exact isomorphism search can visit."""

    states = 1
    falling_factorial = 1
    for depth in range(1, order + 1):
        falling_factorial *= order - depth + 1
        states += falling_factorial
    return states


def _per_candidate_work(pattern_order: int) -> int:
    if pattern_order == 0:
        return 0
    # Creating the subset tuple and the explicit local graph each visits every
    # selected vertex. Every possible local edge incurs one O(1) host-edge
    # probe and, in the dense case, one O(1) insertion in the candidate graph.
    subset_and_vertex_construction = 2 * pattern_order
    pair_probes_and_edge_insertions = 2 * max(1, comb(pattern_order, 2))
    # Edge-count and degree iteration visit the local vertices, while sorting
    # the degree sequence is conservatively bounded by order^2 comparisons.
    local_edge_and_degree_checks = 2 * pattern_order
    degree_sort_checks = max(1, pattern_order * pattern_order)
    # VF2++ explores a subset of the partial injective maps. Candidate
    # generation and feasibility inspect at most order^2 adjacency relations
    # at each such state, including its preprocessing and degree checks.
    isomorphism_checks = _partial_injection_state_bound(pattern_order) * max(
        1, pattern_order * pattern_order
    )
    return (
        subset_and_vertex_construction
        + pair_probes_and_edge_insertions
        + local_edge_and_degree_checks
        + degree_sort_checks
        + isomorphism_checks
    )


def _encode_with_bound(
    payload: object,
    *,
    limit: int,
    error: str,
) -> int:
    try:
        encoded = encode_strict_json(
            payload,
            limits=CanonicalLimits(max_output_bytes=limit),
        )
    except CanonicalizationError as exc:
        raise PydanticCustomError(
            "graph.error_from_exc_return_len_encoded", error
        ) from exc
    return len(encoded)


def _require_bounded_request(
    host: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
) -> None:
    limits = CanonicalLimits()
    host_payload = host.model_dump(mode="json")
    pattern_payload = pattern.model_dump(mode="json")
    source_payload = {
        "host": host_payload,
        "pattern": pattern_payload,
    }
    source_bytes = _encode_with_bound(
        source_payload,
        limit=limits.max_input_bytes,
        error=(
            "host and pattern exceed the canonical "
            f"{limits.max_input_bytes}-byte input bound"
        ),
    )

    host_order = len(host.vertices)
    pattern_order = len(pattern.vertices)
    candidate_count = _candidate_subset_count(host_order, pattern_order)
    if candidate_count > MAX_INDUCED_PATTERN_SUBSETS_PER_PASS:
        raise PydanticCustomError(
            "graph.induced_pattern_subset_count_candidate_count_exceeds",
            f"induced-pattern subset count {candidate_count:,} exceeds the "
            f"{MAX_INDUCED_PATTERN_SUBSETS_PER_PASS:,}-candidate bound",
        )

    maximum_count = format_canonical_integer(candidate_count)
    result_payload = {
        "host": host_payload,
        "pattern": pattern_payload,
        "occurrence_count": maximum_count,
    }
    result_bytes = _encode_with_bound(
        result_payload,
        limit=limits.max_output_bytes,
        error=(
            "the induced-pattern count result retains host and pattern and exceeds "
            f"the canonical {limits.max_output_bytes}-byte output bound"
        ),
    )

    graph_records = (
        len(host.vertices)
        + len(host.edges)
        + len(pattern.vertices)
        + len(pattern.edges)
    )
    total_work = COUNT_PASSES * (
        source_bytes
        + result_bytes
        + graph_records
        + max(1, pattern_order * pattern_order)
        + candidate_count * _per_candidate_work(pattern_order)
    )
    if total_work > MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:
        raise PydanticCustomError(
            "graph.induced_pattern_exact_count_requires_total_work",
            f"induced-pattern exact count requires {total_work:,} work units, "
            f"exceeding the {MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,}-unit bound "
            "for graph construction, direct host-edge probes, explicit candidate "
            "scans, and VF2++ search",
        )


class InducedVertexSubsetPatternCountRequest(StrictModel):
    """Count vertex subsets whose induced graph is isomorphic to a pattern."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Count vertex subsets S of the canonical host for which host[S] "
                "is isomorphic to the canonical pattern. Graphs use the shared "
                "SimpleUndirectedGraph bounds (at most 256 vertices and 32,640 "
                "edges). Admission preflights the exact candidate count "
                "C(|V(host)|, |V(pattern)|), encoded request and retained-result "
                "bytes, graph records, explicit candidate construction from "
                "C(|V(pattern)|, 2) direct host-edge probes per subset, local "
                "candidate scans, and a worst-case VF2++ partial-injection state "
                "bound for every subset. It permits at most 5,000 subsets per pass "
                "and 64,000,000 total work units for the one exact count. "
                "The retained result must fit the "
                "10,485,760-byte canonical output bound. These are conservative "
                "current-backend limits, not restrictions on the mathematical "
                "definition."
            )
        }
    )

    host: SimpleUndirectedGraph = Field(
        description=(
            "Canonical finite simple undirected host graph; vertex labels are "
            "transport and do not constrain isomorphism. Output from "
            "explicit_graph or compose_graphs is accepted unchanged."
        )
    )
    pattern: SimpleUndirectedGraph = Field(
        description=(
            "Canonical finite simple undirected pattern graph. An empty pattern "
            "has one occurrence; a pattern larger than the host has zero. Output "
            "from explicit_graph or compose_graphs is accepted unchanged."
        )
    )

    @model_validator(mode="after")
    def require_complete_exact_count_bounded(self) -> Self:
        _require_bounded_request(self.host, self.pattern)
        return self


class InducedVertexSubsetPatternCountResult(StrictModel):
    """Exact induced-pattern subset count bound to both source graphs.

    The trusted count kernel constructs this value.  The explicit owner
    verifier replays the bounded count only for an independently supplied
    claim.
    """

    host: SimpleUndirectedGraph
    pattern: SimpleUndirectedGraph
    occurrence_count: CanonicalInteger = Field(
        min_length=1,
        max_length=MAX_INDUCED_PATTERN_COUNT_DIGITS,
        description=(
            "Canonical nonnegative decimal count of host vertex subsets inducing "
            "a graph isomorphic to pattern; subsets, not labelled maps, are counted."
        ),
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        _require_bounded_request(self.host, self.pattern)
        claimed = parse_canonical_integer(self.occurrence_count)
        if claimed < 0:
            raise PydanticCustomError(
                "graph.occurrence_count_must_be_nonnegative",
                "occurrence_count must be nonnegative",
            )

        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        host: SimpleUndirectedGraph,
        pattern: SimpleUndirectedGraph,
        occurrence_count: CanonicalInteger,
    ) -> Self:
        """Construct a count emitted by the trusted owner-local kernel."""

        return cls.model_construct(
            host=host,
            pattern=pattern,
            occurrence_count=occurrence_count,
        )


__all__ = [
    "InducedVertexSubsetPatternCountRequest",
    "InducedVertexSubsetPatternCountResult",
]
