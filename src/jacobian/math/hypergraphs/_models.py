"""Typed wire contracts for finite hypergraph operations."""

from __future__ import annotations

import unicodedata
from typing import Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import CanonicalLimits, encode_strict_json, sha256_digest
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 100
MAX_EDGES = 100
MAX_LABEL_LENGTH = 64
MAX_SOLVER_CALLS = MAX_VERTICES

HypergraphIndependenceStatus = Literal["EXACT", "UNKNOWN"]
HypergraphIndependenceTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "WALL_TIME",
    "SOLVER_CALL_LIMIT",
    "SOLVER_ERROR",
    "SOLVER_UNKNOWN",
    "SPECIAL_CASE",
]

MAX_TOTAL_INCIDENCES = MAX_VERTICES * MAX_EDGES
MAX_EDGE_PAIR_COUNT = MAX_EDGES * (MAX_EDGES - 1) // 2
MAX_EDGE_INTERSECTION_CELLS = 65_536
# UTF-8 uses at most four bytes per Unicode scalar.  Together with the existing
# per-label and vertex/edge-count limits, this is the complete aggregate label
# envelope for a FiniteHypergraph source.
MAX_HYPERGRAPH_LABEL_BYTES = (MAX_VERTICES + MAX_EDGES) * MAX_LABEL_LENGTH * 4
MAX_EDGE_INTERSECTION_RESULT_BYTES = CanonicalLimits().max_output_bytes

# One pair entry's keys, punctuation, array brackets, commas, and bounded
# integer occupy fewer than 128 bytes beyond its encoded labels.  The root
# reserve covers the pair-array wrapper, histogram (at most 101 small integer
# pairs), scalar fields, and retained-source field name.
_PAIR_ENTRY_OVERHEAD_BYTES = 128
_RESULT_ENVELOPE_RESERVE_BYTES = 4_096


def _validation_error(message: str) -> PydanticCustomError:
    """Return a structured, actionable owner-local validation reason."""

    # Keep the public explanation readable while assigning stable semantic
    # codes.  The final branch is deliberately fail-closed: adding a new
    # validator requires adding its reason here rather than creating an opaque
    # catch-all error type.
    reason_fragments = (
        ("valid UTF-8", "label_encoding"),
        ("label exceeds", "label_length"),
        ("labels must be distinct", "vertex_identity"),
        ("edge id", "edge_identity"),
        ("edge members", "edge_members"),
        ("declared vertex", "edge_members"),
        ("empty hyperedge", "empty_edge"),
        ("empty edges", "empty_edge"),
        ("digest", "source_digest"),
        ("incumbent vertices", "witness_order"),
        ("incumbent witness", "witness_independence"),
        ("feasible incumbent", "lower_bound"),
        ("solver calls", "solver_budget"),
        ("bounds must lie", "bounds_range"),
        ("upper bound failed", "upper_bound_replay"),
        ("exact result must", "exact_optimum"),
        ("exact result cannot", "wall_budget"),
        ("source-trivial", "special_case"),
        ("exact optimum", "exact_optimum"),
        ("solver-call count", "solver_calls"),
        ("special-case exactness", "special_case"),
        ("incomplete termination", "termination_reason"),
        ("incomplete search", "unknown_result"),
        ("nontrivial bound", "unknown_result"),
        ("solver-call termination", "termination_reason"),
        ("wall-time termination", "termination_reason"),
        ("solver-unknown termination", "termination_reason"),
        ("solver-error termination", "termination_reason"),
        ("unknown result", "termination_reason"),
        ("aggregate", "preflight_aggregate"),
        ("pair bound", "preflight_pairs"),
        ("incidence bound", "preflight_incidences"),
        ("intersection-cell bound", "preflight_cells"),
        ("canonical output limit", "preflight_output"),
        ("distinct edge IDs", "entry_identity"),
        ("intersection_size", "entry_size"),
        ("intersection vertices", "entry_order"),
        ("canonical source ledger", "ledger"),
        ("intersection-size histogram", "histogram"),
        ("edge-pair count", "pair_count"),
        ("maximum_intersection_size", "maximum_intersection"),
        ("is_linear", "linearity"),
        ("first_linearity_violation", "linearity"),
        ("exact number of vertices", "parameters"),
        ("exact number of edges", "parameters"),
        ("exact maximum edge size", "parameters"),
        ("exact minimum edge size", "parameters"),
        ("exact uniformity", "parameters"),
        ("exact incidence count", "parameters"),
        ("vertex-degree", "degrees"),
        ("degree histogram", "degrees"),
        ("exact dual", "dual"),
        ("vertex-to-edges", "incidence_graph"),
        ("edge-to-vertices", "incidence_graph"),
        ("exact incidence pairs", "incidence_graph"),
        ("NFC-normalized", "nfc_labels"),
        ("2-section", "clique_expansion"),
    )
    for fragment, reason in reason_fragments:
        if fragment in message:
            return PydanticCustomError(f"hypergraph.{reason}", message)
    raise AssertionError(f"unmapped hypergraph validation reason: {message}")


def _encoded_utf8_label(label: str) -> bytes:
    """Return the label's UTF-8 encoding or reject unencodable labels.

    Unpaired surrogate code points cannot appear in strict JSON text or in
    the RFC 8785 digest of a source hypergraph, so they are outside the
    admitted label domain.
    """

    try:
        return label.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _validation_error("hypergraph labels must be valid UTF-8") from exc


class FiniteHypergraph(StrictModel):
    """A finite hypergraph: a finite set of vertices and named hyperedges.

    ``vertices`` is a tuple of unique string labels.  ``edges`` is a tuple
    of ``(edge_id, vertex_subset)`` pairs where ``vertex_subset`` is a tuple
    of vertex labels.  Edge member order is irrelevant and is canonicalized
    to sorted order on construction, so two hypergraphs with the same
    members in different orders compare equal.  Every edge member must be a
    declared vertex.  Vertex labels and edge IDs must encode as UTF-8;
    strings containing unpaired surrogate code points are rejected.
    """

    vertices: tuple[str, ...] = Field(
        max_length=MAX_VERTICES,
        description=(
            "Unique UTF-8-encodable vertex labels of at most "
            f"{MAX_LABEL_LENGTH} characters each."
        ),
    )
    edges: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        max_length=MAX_EDGES,
        description=(
            "Unique UTF-8-encodable edge IDs (at most "
            f"{MAX_LABEL_LENGTH} characters each) paired with tuples of "
            "declared vertex labels."
        ),
    )

    @model_validator(mode="after")
    def require_valid_hypergraph(self) -> Self:
        labels = set(self.vertices)
        if len(labels) != len(self.vertices):
            raise _validation_error("vertex labels must be distinct")
        for label in self.vertices:
            if len(label) > MAX_LABEL_LENGTH:
                raise _validation_error(
                    "vertex label exceeds the bounded length budget"
                )
            _encoded_utf8_label(label)
        edge_ids: set[str] = set()
        canonical_edges: list[tuple[str, tuple[str, ...]]] = []
        for edge_id, members in self.edges:
            if len(edge_id) > MAX_LABEL_LENGTH:
                raise _validation_error("edge id exceeds the bounded length budget")
            _encoded_utf8_label(edge_id)
            if edge_id in edge_ids:
                raise _validation_error("edge ids must be distinct")
            edge_ids.add(edge_id)
            member_set = set(members)
            if len(member_set) != len(members):
                raise _validation_error("edge members must be distinct")
            unknown = member_set - labels
            if unknown:
                raise _validation_error("every edge member must be a declared vertex")
            canonical_edges.append((edge_id, tuple(sorted(members))))
        object.__setattr__(self, "edges", tuple(canonical_edges))
        return self


def _hypergraph_digest(hypergraph: FiniteHypergraph) -> str:
    payload = {
        "format": "jacobian.finite-hypergraph/v1",
        "vertices": list(hypergraph.vertices),
        "edges": [[edge_id, list(members)] for edge_id, members in hypergraph.edges],
    }
    return sha256_digest(encode_strict_json(payload))


def _greedy_independent_vertices(
    hypergraph: FiniteHypergraph,
) -> tuple[str, ...]:
    """Return one deterministic feasible incumbent in declared vertex order."""

    edge_sets = tuple(frozenset(members) for _, members in hypergraph.edges)
    selected: set[str] = set()
    for vertex in hypergraph.vertices:
        candidate = selected | {vertex}
        if not any(edge <= candidate for edge in edge_sets):
            selected.add(vertex)
    return tuple(vertex for vertex in hypergraph.vertices if vertex in selected)


def _independence_upper_bound(hypergraph: FiniteHypergraph) -> int:
    forbidden_vertices = {
        members[0] for _, members in hypergraph.edges if len(members) == 1
    }
    return len(hypergraph.vertices) - len(forbidden_vertices)


class HypergraphIndependenceBudget(StrictModel):
    """Explicit wall-time and solver-call bounds for one exact search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)
    max_solver_calls: StrictInt = Field(
        default=MAX_SOLVER_CALLS,
        ge=1,
        le=MAX_SOLVER_CALLS,
        description=(
            "Maximum monotone cardinality thresholds submitted during search. "
            "Independent validation of an externally supplied result may replay "
            "one additional upper-bound threshold."
        ),
    )


class HypergraphIndependenceRequest(StrictModel):
    """One finite hypergraph and its operation-owned exact-search budget.

    Hyperedges must be nonempty. The inherited finite-hypergraph contract admits
    at most 100 vertices, 100 indexed edges, width 100, and 10,000 total
    incidences, which bounds the Boolean encoding before the private backend is
    invoked.
    """

    hypergraph: FiniteHypergraph
    resource_budget: HypergraphIndependenceBudget = Field(
        default_factory=HypergraphIndependenceBudget
    )

    @model_validator(mode="after")
    def require_supported_encoding(self) -> Self:
        if any(not members for _, members in self.hypergraph.edges):
            raise _validation_error(
                "independence-number search does not admit empty edges"
            )
        return self


# Internal-only validation-context key. The producer-side constructor in
# ``_independence_z3`` sets it after its own threshold search has proved every
# reported bound within the same bounded call, skipping only the duplicate
# upper-bound solver replay during construction. Independently supplied results
# never carry this key and always execute the bounded replay.
_PRODUCER_ESTABLISHED_BOUNDS = "producer_established_bounds"


class HypergraphIndependenceResult(StrictModel):
    """Exact optimum or sound incumbent and bounds for one source hypergraph."""

    hypergraph: FiniteHypergraph
    hypergraph_digest: Sha256Digest
    resource_budget: HypergraphIndependenceBudget
    status: HypergraphIndependenceStatus
    independence_number: StrictInt | None = Field(default=None, ge=0, le=MAX_VERTICES)
    incumbent_vertices: tuple[str, ...] = Field(max_length=MAX_VERTICES)
    lower_bound: StrictInt = Field(ge=0, le=MAX_VERTICES)
    upper_bound: StrictInt = Field(ge=0, le=MAX_VERTICES)
    solver_calls: StrictInt = Field(ge=0, le=MAX_SOLVER_CALLS)
    wall_budget_exhausted: StrictBool
    termination_reason: HypergraphIndependenceTermination
    detail: str = Field(min_length=1, max_length=1024)
    convention: Literal["MAXIMUM_NO_COMPLETE_HYPEREDGE_VERTEX_SUBSET"] = (
        "MAXIMUM_NO_COMPLETE_HYPEREDGE_VERTEX_SUBSET"
    )

    @model_validator(mode="after")
    def bind_source_and_witness(self) -> Self:
        if any(not members for _, members in self.hypergraph.edges):
            raise _validation_error("result source must not contain an empty hyperedge")
        if self.hypergraph_digest != _hypergraph_digest(self.hypergraph):
            raise _validation_error(
                "hypergraph_digest must bind the exact source hypergraph"
            )

        witness_set = set(self.incumbent_vertices)
        expected_order = tuple(
            vertex for vertex in self.hypergraph.vertices if vertex in witness_set
        )
        if self.incumbent_vertices != expected_order or len(witness_set) != len(
            self.incumbent_vertices
        ):
            raise _validation_error(
                "incumbent vertices must be unique and in declared vertex order"
            )
        if any(set(members) <= witness_set for _, members in self.hypergraph.edges):
            raise _validation_error(
                "incumbent witness must contain no complete hyperedge"
            )
        if self.lower_bound != len(self.incumbent_vertices):
            raise _validation_error("the feasible incumbent must be the lower bound")
        return self

    @model_validator(mode="after")
    def bind_bounds_to_source(self, info: ValidationInfo) -> Self:
        initial_upper = _independence_upper_bound(self.hypergraph)
        if self.solver_calls > self.resource_budget.max_solver_calls:
            raise _validation_error("solver calls must fit the submitted call budget")
        if not self.lower_bound <= self.upper_bound <= initial_upper:
            raise _validation_error(
                "independence-number bounds must lie in the source range"
            )
        producer_established = (info.context or {}).get(_PRODUCER_ESTABLISHED_BOUNDS)
        if self.upper_bound < initial_upper and not producer_established:
            from jacobian.math.hypergraphs import _independence_z3

            if not _independence_z3.verify_upper_bound(
                self.hypergraph,
                self.upper_bound,
                self.resource_budget.wall_seconds,
            ):
                raise _validation_error("upper bound failed its bounded source replay")
        return self

    @model_validator(mode="after")
    def bind_exact_completion(self) -> Self:
        if self.status != "EXACT":
            return self
        initial_incumbent = _greedy_independent_vertices(self.hypergraph)
        initial_upper = _independence_upper_bound(self.hypergraph)
        if (
            self.independence_number is None
            or self.independence_number != self.lower_bound
            or self.independence_number != self.upper_bound
        ):
            raise _validation_error("exact result must bind one coincident optimum")
        if self.wall_budget_exhausted:
            raise _validation_error("an exact result cannot exhaust its wall budget")
        if self.termination_reason == "OPTIMUM_ESTABLISHED":
            if len(initial_incumbent) == initial_upper:
                raise _validation_error(
                    "a source-trivial optimum must use SPECIAL_CASE"
                )
            if self.independence_number < len(initial_incumbent):
                raise _validation_error(
                    "an exact optimum cannot be below a feasible witness"
                )
            expected_calls = initial_upper - self.independence_number
            if self.independence_number > len(initial_incumbent):
                expected_calls += 1
            if self.solver_calls != expected_calls:
                raise _validation_error(
                    "exact solver-call count must match the descending thresholds"
                )
        elif self.termination_reason == "SPECIAL_CASE":
            if self.solver_calls != 0 or len(initial_incumbent) != initial_upper:
                raise _validation_error(
                    "special-case exactness requires coincident initial bounds"
                )
        else:
            raise _validation_error("exact result has an incomplete termination reason")
        return self

    @model_validator(mode="after")
    def bind_unknown_completion(self) -> Self:
        if self.status != "UNKNOWN":
            return self
        if self.independence_number is not None:
            raise _validation_error(
                "incomplete search cannot claim an independence number"
            )
        if self.lower_bound >= self.upper_bound:
            raise _validation_error("unknown result must retain a nontrivial bound gap")
        initial_upper = _independence_upper_bound(self.hypergraph)
        proved_thresholds = initial_upper - self.upper_bound
        if self.termination_reason == "SOLVER_CALL_LIMIT":
            if (
                self.wall_budget_exhausted
                or self.solver_calls != self.resource_budget.max_solver_calls
                or proved_thresholds != self.solver_calls
            ):
                raise _validation_error(
                    "solver-call termination must exhaust exactly its query budget"
                )
        elif self.termination_reason == "WALL_TIME":
            if not self.wall_budget_exhausted or proved_thresholds not in {
                self.solver_calls,
                max(0, self.solver_calls - 1),
            }:
                raise _validation_error(
                    "wall-time termination must bind the completed thresholds"
                )
        elif self.termination_reason == "SOLVER_UNKNOWN":
            if (
                self.wall_budget_exhausted
                or self.solver_calls == 0
                or proved_thresholds != self.solver_calls - 1
            ):
                raise _validation_error(
                    "solver-unknown termination must bind its inconclusive query"
                )
        elif self.termination_reason == "SOLVER_ERROR":
            if self.wall_budget_exhausted or self.upper_bound != initial_upper:
                raise _validation_error(
                    "solver-error termination must retain the source bound"
                )
        else:
            raise _validation_error("unknown result has an exact termination reason")
        return self


def _label_utf8_bytes(label: str) -> int:
    return len(_encoded_utf8_label(label))


def _strict_label_wire_bytes(label: str) -> int:
    """Return the exact encoded size used by the public result wrapper."""

    return len(encode_strict_json(label))


def _edge_intersection_preflight_data(
    hypergraph: FiniteHypergraph,
) -> tuple[int, int, int, int]:
    """Return pair, incidence, intersection-cell, and result-byte quantities.

    The result estimate is computed without intersecting or materializing any
    edge pair.  A vertex of degree ``d`` occurs in exactly ``C(d, 2)`` pair
    intersections, so incidence degrees give both the exact returned-cell
    count and the exact encoded contribution of all intersection labels.
    """

    edge_count = len(hypergraph.edges)
    pair_count = edge_count * (edge_count - 1) // 2
    vertex_degrees = dict.fromkeys(hypergraph.vertices, 0)
    total_incidences = 0
    for _, members in hypergraph.edges:
        total_incidences += len(members)
        for member in members:
            vertex_degrees[member] += 1
    vertex_pair_multiplicities = {
        vertex: degree * (degree - 1) // 2 for vertex, degree in vertex_degrees.items()
    }
    intersection_cells = sum(vertex_pair_multiplicities.values())

    labels = (*hypergraph.vertices, *(edge_id for edge_id, _ in hypergraph.edges))
    label_bytes = sum(_label_utf8_bytes(label) for label in labels)
    if label_bytes > MAX_HYPERGRAPH_LABEL_BYTES:
        raise _validation_error(
            "hypergraph labels exceed the aggregate "
            f"{MAX_HYPERGRAPH_LABEL_BYTES}-byte UTF-8 bound"
        )

    vertex_wire_bytes = {
        vertex: _strict_label_wire_bytes(vertex) for vertex in hypergraph.vertices
    }
    edge_id_wire_bytes = tuple(
        _strict_label_wire_bytes(edge_id) for edge_id, _ in hypergraph.edges
    )
    # Add one byte per returned member for a comma.  This overcounts each
    # nonempty JSON array by one byte and therefore remains a safe bound.
    pair_intersection_bytes = sum(
        multiplicity * (vertex_wire_bytes[vertex] + 1)
        for vertex, multiplicity in vertex_pair_multiplicities.items()
    )
    pair_label_bytes = (edge_count - 1) * sum(edge_id_wire_bytes)
    largest_edge_id_bytes = sum(sorted(edge_id_wire_bytes, reverse=True)[:2])
    possible_violation_members = sum(
        vertex_wire_bytes[vertex] + 1
        for vertex, degree in vertex_degrees.items()
        if degree >= 2
    )
    maximum_pair_payload_bytes = largest_edge_id_bytes + possible_violation_members

    source_bytes = len(encode_strict_json(hypergraph.model_dump(mode="json")))
    estimated_result_bytes = (
        source_bytes
        + pair_count * _PAIR_ENTRY_OVERHEAD_BYTES
        + pair_label_bytes
        + pair_intersection_bytes
        # A nonlinear result repeats the first complete violating entry.
        + (_PAIR_ENTRY_OVERHEAD_BYTES + maximum_pair_payload_bytes if pair_count else 0)
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    return pair_count, total_incidences, intersection_cells, estimated_result_bytes


def _require_edge_intersection_preflight(hypergraph: FiniteHypergraph) -> None:
    (
        pair_count,
        total_incidences,
        intersection_cells,
        estimated_result_bytes,
    ) = _edge_intersection_preflight_data(hypergraph)
    if pair_count > MAX_EDGE_PAIR_COUNT:
        raise _validation_error(
            f"edge-intersection profile exceeds the {MAX_EDGE_PAIR_COUNT}-pair bound"
        )
    if total_incidences > MAX_TOTAL_INCIDENCES:
        raise _validation_error(
            "edge-intersection source exceeds the "
            f"{MAX_TOTAL_INCIDENCES}-incidence bound"
        )
    if intersection_cells > MAX_EDGE_INTERSECTION_CELLS:
        raise _validation_error(
            "edge-intersection profile exceeds the "
            f"{MAX_EDGE_INTERSECTION_CELLS}-intersection-cell bound"
        )
    if estimated_result_bytes > MAX_EDGE_INTERSECTION_RESULT_BYTES:
        raise _validation_error(
            "the complete edge-intersection profile would exceed the "
            f"{MAX_EDGE_INTERSECTION_RESULT_BYTES}-byte canonical output limit; "
            "shorten labels or reduce the edge family"
        )


class EdgeIntersectionsRequest(StrictModel):
    """Request the complete indexed edge-intersection profile.

    Every unordered pair of distinct indexed edges is returned.  Admission
    bounds the pair family, aggregate source incidences, exact returned
    intersection memberships, label bytes, and canonical serialized result
    before any pair intersection is materialized.
    """

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Canonical finite hypergraph with at most 100 vertices and 100 "
            "indexed edges. The complete profile is admitted only when its "
            "pair intersections contain at most 65,536 memberships "
            "and its retained-source result fits the canonical output limit."
        ),
        json_schema_extra={
            "edge_pair_bound": MAX_EDGE_PAIR_COUNT,
            "aggregate_input_incidences_bound": MAX_TOTAL_INCIDENCES,
            "intersection_cells_bound": MAX_EDGE_INTERSECTION_CELLS,
            "aggregate_label_bytes_bound": MAX_HYPERGRAPH_LABEL_BYTES,
            "canonical_result_bytes_bound": MAX_EDGE_INTERSECTION_RESULT_BYTES,
        },
    )

    @model_validator(mode="after")
    def require_bounded_complete_profile(self) -> Self:
        _require_edge_intersection_preflight(self.hypergraph)
        return self


class EdgeIntersectionEntry(StrictModel):
    """One canonical unordered indexed-edge pair and its exact intersection."""

    left_edge_id: str = Field(max_length=MAX_LABEL_LENGTH)
    right_edge_id: str = Field(max_length=MAX_LABEL_LENGTH)
    intersection: tuple[str, ...] = Field(max_length=MAX_VERTICES)
    intersection_size: int = Field(ge=0, le=MAX_VERTICES)

    @model_validator(mode="after")
    def require_canonical_entry(self) -> Self:
        if self.left_edge_id == self.right_edge_id:
            raise _validation_error(
                "an edge-intersection pair must use distinct edge IDs"
            )
        if self.intersection_size != len(self.intersection):
            raise _validation_error(
                "intersection_size must equal the number of intersection vertices"
            )
        if self.intersection != tuple(sorted(set(self.intersection))):
            raise _validation_error(
                "intersection vertices must be distinct and lexicographically sorted"
            )
        return self


class EdgeIntersectionsResult(StrictModel):
    """Complete source-bound indexed edge-intersection profile.

    ``pair_intersections`` contains every unordered pair exactly once in
    declared edge order.  ``histogram`` is reconstructed from that ledger.
    The maximum, linearity decision, and first canonical violation are derived
    from the same authoritative entries and replayed against ``hypergraph``.
    """

    hypergraph: FiniteHypergraph
    pair_intersections: tuple[EdgeIntersectionEntry, ...] = Field(
        max_length=MAX_EDGE_PAIR_COUNT
    )
    pair_count: int = Field(ge=0, le=MAX_EDGE_PAIR_COUNT)
    histogram: tuple[tuple[int, int], ...] = Field(max_length=MAX_VERTICES + 1)
    maximum_intersection_size: int = Field(ge=0, le=MAX_VERTICES)
    is_linear: bool
    first_linearity_violation: EdgeIntersectionEntry | None = None

    @model_validator(mode="before")
    @classmethod
    def require_aggregate_intersection_bound(cls, data: object) -> object:
        """Reject an oversized authored ledger before nested model parsing."""

        data = canonicalize_json_containers(data)

        if not isinstance(data, dict):
            return data
        entries = data.get("pair_intersections")
        if not isinstance(entries, (list, tuple)):
            return data
        total = 0
        for entry in entries:
            if isinstance(entry, EdgeIntersectionEntry):
                total += len(entry.intersection)
            elif isinstance(entry, dict):
                intersection = entry.get("intersection")
                if isinstance(intersection, (list, tuple)):
                    total += len(intersection)
            if total > MAX_EDGE_INTERSECTION_CELLS:
                raise _validation_error(
                    "the aggregate returned intersections exceed the "
                    f"{MAX_EDGE_INTERSECTION_CELLS}-cell bound"
                )
        return data

    @model_validator(mode="after")
    def bind_edge_intersections(self) -> Self:
        from jacobian.math.hypergraphs._operations import _edge_intersections_data

        _require_edge_intersection_preflight(self.hypergraph)
        (
            pair_intersections,
            histogram,
            pair_count,
            maximum_intersection_size,
            is_linear,
            first_linearity_violation,
        ) = _edge_intersections_data(self.hypergraph)
        if self.pair_intersections != pair_intersections:
            raise _validation_error(
                "pair_intersections must be the complete canonical source ledger"
            )
        if self.histogram != histogram:
            raise _validation_error(
                "histogram must be the exact intersection-size histogram"
            )
        if self.pair_count != pair_count:
            raise _validation_error(
                "pair_count must equal the complete edge-pair count"
            )
        if self.maximum_intersection_size != maximum_intersection_size:
            raise _validation_error(
                "maximum_intersection_size must be derived from pair_intersections"
            )
        if self.is_linear != is_linear:
            raise _validation_error("is_linear must match the exact pair intersections")
        if self.first_linearity_violation != first_linearity_violation:
            raise _validation_error(
                "first_linearity_violation must be the first canonical pair "
                "whose intersection has size greater than one"
            )
        return self


class ParametersRequest(StrictModel):
    """Request the basic parameters of a finite hypergraph."""

    hypergraph: FiniteHypergraph


class ParametersResult(StrictModel):
    """The basic parameters of a finite hypergraph.

    ``vertex_count`` and ``edge_count`` are the number of vertices and edges.
    ``rank`` is the size of the largest edge, ``corank`` the size of the
    smallest edge, and ``uniform_size`` is that common edge size when every
    edge has the same cardinality (``None`` otherwise).
    ``total_incidences`` is the sum of all edge cardinalities.
    """

    hypergraph: FiniteHypergraph
    vertex_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    rank: int = Field(ge=0)
    corank: int = Field(ge=0)
    uniform_size: int | None = None
    total_incidences: int = Field(ge=0)

    @model_validator(mode="after")
    def bind_parameters(self) -> Self:
        from jacobian.math.hypergraphs._operations import _parameters_data

        (
            vertex_count,
            edge_count,
            rank,
            corank,
            uniform_size,
            total_incidences,
        ) = _parameters_data(self.hypergraph)
        if self.vertex_count != vertex_count:
            raise _validation_error("vertex_count must be the exact number of vertices")
        if self.edge_count != edge_count:
            raise _validation_error("edge_count must be the exact number of edges")
        if self.rank != rank:
            raise _validation_error("rank must be the exact maximum edge size")
        if self.corank != corank:
            raise _validation_error("corank must be the exact minimum edge size")
        if self.uniform_size != uniform_size:
            raise _validation_error("uniform_size must match the exact uniformity")
        if self.total_incidences != total_incidences:
            raise _validation_error(
                "total_incidences must be the exact incidence count"
            )
        return self


class VertexDegreesRequest(StrictModel):
    """Request the vertex-degree map of a finite hypergraph."""

    hypergraph: FiniteHypergraph


class VertexDegreesResult(StrictModel):
    """The vertex-degree map and degree histogram of a finite hypergraph.

    ``degrees`` maps each vertex label to its degree (the number of edges
    containing it), in declared vertex order.  ``histogram`` maps each
    degree value to the number of vertices with that degree, sorted by
    degree ascending.
    """

    hypergraph: FiniteHypergraph
    degrees: tuple[tuple[str, int], ...]
    histogram: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def bind_vertex_degrees(self) -> Self:
        from jacobian.math.hypergraphs._operations import _vertex_degrees_data

        degrees, histogram = _vertex_degrees_data(self.hypergraph)
        if self.degrees != degrees:
            raise _validation_error(
                "degrees must be the exact vertex-degree map of the hypergraph"
            )
        if self.histogram != histogram:
            raise _validation_error(
                "histogram must be the exact degree histogram of the hypergraph"
            )
        return self


class DualRequest(StrictModel):
    """Request the dual of a finite hypergraph."""

    hypergraph: FiniteHypergraph


class DualResult(StrictModel):
    """The dual of a finite hypergraph.

    The dual hypergraph transposes vertices and edges: the original edges
    become vertices and the original vertices become edges, where vertex
    ``v`` becomes the edge containing one dual vertex ``e`` for each original
    edge containing ``v``.
    """

    hypergraph: FiniteHypergraph
    dual: FiniteHypergraph

    @model_validator(mode="after")
    def bind_dual(self) -> Self:
        from jacobian.math.hypergraphs._operations import _dual_data

        dual = _dual_data(self.hypergraph)
        if self.dual != dual:
            raise _validation_error("dual must be the exact dual hypergraph")
        return self


class IncidenceGraphRequest(StrictModel):
    """Request the bipartite incidence graph (Levi graph) of a hypergraph."""

    hypergraph: FiniteHypergraph


class IncidenceGraphResult(StrictModel):
    """The bipartite incidence graph (Levi graph) of a finite hypergraph.

    ``vertex_incidence`` maps each vertex label to the tuple of edge ids
    containing it, in declared edge order.  ``edge_incidence`` maps each
    edge id to the tuple of vertex labels it contains, in sorted member
    order (the canonical edge order).  ``edges`` lists the
    ``(vertex, edge_id)`` incidence pairs, sorted by vertex in declared
    order then by edge id.
    """

    hypergraph: FiniteHypergraph
    vertex_incidence: tuple[tuple[str, tuple[str, ...]], ...]
    edge_incidence: tuple[tuple[str, tuple[str, ...]], ...]
    edges: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def bind_incidence_graph(self) -> Self:
        from jacobian.math.hypergraphs._operations import _incidence_graph_data

        vertex_incidence, edge_incidence, edges = _incidence_graph_data(self.hypergraph)
        if self.vertex_incidence != vertex_incidence:
            raise _validation_error(
                "vertex_incidence must be the exact vertex-to-edges map"
            )
        if self.edge_incidence != edge_incidence:
            raise _validation_error(
                "edge_incidence must be the exact edge-to-vertices map"
            )
        if self.edges != edges:
            raise _validation_error("edges must be the exact incidence pairs")
        return self


class CliqueExpansionRequest(StrictModel):
    """Request the clique expansion (2-section) of a hypergraph.

    The expansion is emitted as the domain-owned canonical
    ``SimpleUndirectedGraph``, whose value requires NFC-normalized vertex
    labels; hypergraphs whose declared labels are not NFC are outside this
    operation's admitted domain.
    """

    hypergraph: FiniteHypergraph

    @model_validator(mode="after")
    def require_nfc_vertex_labels(self) -> Self:
        if any(
            not unicodedata.is_normalized("NFC", vertex)
            for vertex in self.hypergraph.vertices
        ):
            raise _validation_error(
                "clique expansion requires NFC-normalized vertex labels"
            )
        return self


class CliqueExpansionResult(StrictModel):
    """The primal/2-section graph of a finite hypergraph.

    ``graph`` is the canonical :class:`SimpleUndirectedGraph` whose vertices
    are exactly the hypergraph's declared vertex labels, in declared order,
    and in which two distinct vertices are adjacent if and only if they
    share at least one hyperedge.  Each edge's endpoints appear in lexical
    order per the canonical graph value's own convention, independent of the
    source hypergraph's declared ordering, so the value composes directly
    with downstream graph operations without translation.  This defining
    property is validated against the retained source hypergraph.
    """

    hypergraph: FiniteHypergraph
    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def bind_clique_expansion(self) -> Self:
        from jacobian.math.hypergraphs._operations import _clique_expansion_graph

        if self.graph != _clique_expansion_graph(self.hypergraph):
            raise _validation_error(
                "graph must be the exact 2-section of the hypergraph"
            )
        return self
