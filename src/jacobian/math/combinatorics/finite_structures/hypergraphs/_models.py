"""Typed wire contracts for finite hypergraph operations."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    encode_strict_json,
    sha256_digest,
    strict_json_object_size,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 256
MAX_EDGES = 12_000
MAX_LABEL_LENGTH = 64
MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS = 16
MAX_HYPERGRAPH_INDEPENDENCE_VERTICES = 100
MAX_HYPERGRAPH_INDEPENDENCE_INCIDENCES = 10_000

HypergraphIndependenceStatus = Literal["EXACT", "UNKNOWN"]
HypergraphIndependenceTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "WALL_TIME",
    "SOLVER_CALL_LIMIT",
    "SOLVER_ERROR",
    "SOLVER_UNKNOWN",
    "SPECIAL_CASE",
]

MAX_TOTAL_INCIDENCES = 36_000
MAX_EDGE_PAIR_COUNT = 65_536
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
        ("source exceeds", "source_representation"),
        ("solver bound", "solver_envelope"),
        ("dual exceeds", "dual_envelope"),
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
        ("induced type profile", "induced_profile"),
        ("transversal", "transversal"),
        ("matching", "matching"),
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
            f"{MAX_LABEL_LENGTH} characters each; a carrier has at most "
            f"{MAX_VERTICES} vertices."
        ),
    )
    edges: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        max_length=MAX_EDGES,
        description=(
            "Unique UTF-8-encodable edge IDs (at most "
            f"{MAX_LABEL_LENGTH} characters each) paired with tuples of "
            "declared vertex labels. A carrier has at most "
            f"{MAX_EDGES} edges and {MAX_TOTAL_INCIDENCES} total incidences."
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
        total_incidences = sum(len(members) for _, members in self.edges)
        if total_incidences > MAX_TOTAL_INCIDENCES:
            raise _validation_error(
                "hypergraph source exceeds the "
                f"{MAX_TOTAL_INCIDENCES}-incidence representation bound"
            )
        try:
            encode_strict_json(
                {
                    "vertices": list(self.vertices),
                    "edges": [
                        [edge_id, list(members)] for edge_id, members in self.edges
                    ],
                }
            )
        except ValueError as exc:
            raise _validation_error(
                "hypergraph source exceeds the canonical JSON representation bound"
            ) from exc
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
        default=MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS,
        ge=1,
        le=MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS,
        description=(
            "Maximum monotone cardinality thresholds submitted during search. "
            "Independent validation of an externally supplied result may replay "
            "one additional upper-bound threshold."
        ),
    )


class HypergraphIndependenceRequest(StrictModel):
    """One finite hypergraph and its operation-owned exact-search budget.

    Hyperedges must be nonempty. The source carrier has its own representation
    envelope; this operation independently admits only Boolean encodings of at
    most 100 vertices and 10,000 incidences before invoking the private backend.
    """

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Canonical finite hypergraph. The Z3 threshold search separately "
            f"admits at most {MAX_HYPERGRAPH_INDEPENDENCE_VERTICES} vertices "
            f"and {MAX_HYPERGRAPH_INDEPENDENCE_INCIDENCES} incidences."
        )
    )
    resource_budget: HypergraphIndependenceBudget = Field(
        default_factory=HypergraphIndependenceBudget
    )


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
    solver_calls: StrictInt = Field(ge=0, le=MAX_HYPERGRAPH_INDEPENDENCE_SOLVER_CALLS)
    wall_budget_exhausted: StrictBool
    termination_reason: HypergraphIndependenceTermination
    detail: str = Field(min_length=1, max_length=1024)
    convention: Literal["MAXIMUM_NO_COMPLETE_HYPEREDGE_VERTEX_SUBSET"] = (
        "MAXIMUM_NO_COMPLETE_HYPEREDGE_VERTEX_SUBSET"
    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        hypergraph: FiniteHypergraph,
        resource_budget: HypergraphIndependenceBudget,
        status: HypergraphIndependenceStatus,
        independence_number: int | None,
        incumbent_vertices: tuple[str, ...],
        upper_bound: int,
        solver_calls: int,
        wall_budget_exhausted: bool,
        termination_reason: HypergraphIndependenceTermination,
        detail: str,
    ) -> Self:
        """Construct an outcome established by the owner-local Z3 kernel."""

        return cls.model_construct(
            hypergraph=hypergraph,
            hypergraph_digest=_hypergraph_digest(hypergraph),
            resource_budget=resource_budget,
            status=status,
            independence_number=independence_number,
            incumbent_vertices=incumbent_vertices,
            lower_bound=len(incumbent_vertices),
            upper_bound=upper_bound,
            solver_calls=solver_calls,
            wall_budget_exhausted=wall_budget_exhausted,
            termination_reason=termination_reason,
            detail=detail,
        )

    @model_validator(mode="after")
    def bind_source_and_witness(self) -> Self:
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
        if self.lower_bound != len(self.incumbent_vertices):
            raise _validation_error("the feasible incumbent must be the lower bound")
        return self

    @model_validator(mode="after")
    def bind_bounds_to_source(self) -> Self:
        if self.solver_calls > self.resource_budget.max_solver_calls:
            raise _validation_error("solver calls must fit the submitted call budget")
        if not self.lower_bound <= self.upper_bound <= len(self.hypergraph.vertices):
            raise _validation_error(
                "independence-number bounds must lie in the source range"
            )
        return self

    @model_validator(mode="after")
    def bind_exact_completion(self) -> Self:
        if self.status != "EXACT":
            return self
        if (
            self.independence_number is None
            or self.independence_number != self.lower_bound
            or self.independence_number != self.upper_bound
        ):
            raise _validation_error("exact result must bind one coincident optimum")
        if self.wall_budget_exhausted:
            raise _validation_error("an exact result cannot exhaust its wall budget")
        if self.termination_reason not in {
            "OPTIMUM_ESTABLISHED",
            "SPECIAL_CASE",
        }:
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
        if self.termination_reason not in {
            "WALL_TIME",
            "SOLVER_CALL_LIMIT",
            "SOLVER_ERROR",
            "SOLVER_UNKNOWN",
        }:
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


def _admit_edge_intersection_profile(hypergraph: FiniteHypergraph) -> None:
    """Admit one complete profile before its owner-local kernel runs."""
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
            "Canonical finite hypergraph. The complete profile is admitted only when its "
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
    from the same authoritative entries.  An explicit owner verifier checks
    the ledger against the retained source hypergraph when required.
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
        edge_ids = tuple(edge_id for edge_id, _ in self.hypergraph.edges)
        if len(edge_ids) * (len(edge_ids) - 1) // 2 > MAX_EDGE_PAIR_COUNT:
            raise _validation_error(
                f"edge-intersection profile exceeds the {MAX_EDGE_PAIR_COUNT}-pair bound"
            )
        if (
            sum(len(members) for _, members in self.hypergraph.edges)
            > MAX_TOTAL_INCIDENCES
        ):
            raise _validation_error(
                "edge-intersection source exceeds the "
                f"{MAX_TOTAL_INCIDENCES}-incidence bound"
            )
        expected_pairs = tuple(
            (edge_ids[left], edge_ids[right])
            for left in range(len(edge_ids))
            for right in range(left + 1, len(edge_ids))
        )
        pairs = tuple(
            (entry.left_edge_id, entry.right_edge_id)
            for entry in self.pair_intersections
        )
        if pairs != expected_pairs:
            raise _validation_error(
                "pair_intersections must be the complete canonical source ledger"
            )
        if any(
            vertex not in self.hypergraph.vertices
            for entry in self.pair_intersections
            for vertex in entry.intersection
        ):
            raise _validation_error(
                "intersection vertices must be declared source vertices"
            )
        histogram_counts: dict[int, int] = {}
        for entry in self.pair_intersections:
            histogram_counts[entry.intersection_size] = (
                histogram_counts.get(entry.intersection_size, 0) + 1
            )
        histogram = tuple(sorted(histogram_counts.items()))
        if self.histogram != histogram:
            raise _validation_error("histogram must be the intersection-size histogram")
        if self.pair_count != len(self.pair_intersections):
            raise _validation_error(
                "pair_count must equal the complete edge-pair count"
            )
        maximum_intersection_size = max(
            (entry.intersection_size for entry in self.pair_intersections), default=0
        )
        if self.maximum_intersection_size != maximum_intersection_size:
            raise _validation_error(
                "maximum_intersection_size must be derived from pair_intersections"
            )
        first_linearity_violation = next(
            (entry for entry in self.pair_intersections if entry.intersection_size > 1),
            None,
        )
        if self.is_linear != (first_linearity_violation is None):
            raise _validation_error("is_linear must match the pair intersections")
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
    vertex_count: int = Field(ge=0, le=MAX_VERTICES)
    edge_count: int = Field(ge=0, le=MAX_EDGES)
    rank: int = Field(ge=0, le=MAX_VERTICES)
    corank: int = Field(ge=0, le=MAX_VERTICES)
    uniform_size: int | None = Field(default=None, ge=0, le=MAX_VERTICES)
    total_incidences: int = Field(ge=0, le=MAX_TOTAL_INCIDENCES)


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
    degrees: tuple[tuple[str, int], ...] = Field(max_length=MAX_VERTICES)
    histogram: tuple[tuple[int, int], ...] = Field(max_length=MAX_VERTICES + 1)

    @model_validator(mode="after")
    def bind_vertex_degrees(self) -> Self:
        if tuple(vertex for vertex, _ in self.degrees) != self.hypergraph.vertices:
            raise _validation_error(
                "degrees must list each source vertex in declared order"
            )
        if any(degree < 0 or degree > MAX_EDGES for _, degree in self.degrees):
            raise _validation_error(
                "vertex-degree values must be within the source bound"
            )
        expected_histogram_counts: dict[int, int] = {}
        for _, degree in self.degrees:
            expected_histogram_counts[degree] = (
                expected_histogram_counts.get(degree, 0) + 1
            )
        if self.histogram != tuple(sorted(expected_histogram_counts.items())):
            raise _validation_error(
                "histogram must be the degree histogram of the returned map"
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
    vertex_incidence: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        max_length=MAX_VERTICES
    )
    edge_incidence: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        max_length=MAX_EDGES
    )
    edges: tuple[tuple[str, str], ...] = Field(max_length=MAX_TOTAL_INCIDENCES)

    @model_validator(mode="after")
    def bind_incidence_graph(self) -> Self:
        if (
            tuple(vertex for vertex, _ in self.vertex_incidence)
            != self.hypergraph.vertices
        ):
            raise _validation_error(
                "vertex_incidence must list source vertices in declared order"
            )
        edge_ids = tuple(edge_id for edge_id, _ in self.hypergraph.edges)
        if tuple(edge_id for edge_id, _ in self.edge_incidence) != edge_ids:
            raise _validation_error(
                "edge_incidence must list source edge ids in declared order"
            )
        vertex_set = set(self.hypergraph.vertices)
        edge_id_set = set(edge_ids)
        if (
            any(
                vertex not in vertex_set or edge_id not in edge_id_set
                for vertex, edge_ids_for_vertex in self.vertex_incidence
                for edge_id in edge_ids_for_vertex
            )
            or any(
                vertex not in vertex_set
                for _, vertices_for_edge in self.edge_incidence
                for vertex in vertices_for_edge
            )
            or any(
                vertex not in vertex_set or edge_id not in edge_id_set
                for vertex, edge_id in self.edges
            )
        ):
            raise _validation_error("incidence entries must use source labels")
        return self


class CliqueExpansionRequest(StrictModel):
    """Request the clique expansion (2-section) of a hypergraph.

    The expansion is emitted as the domain-owned canonical
    ``SimpleUndirectedGraph``, whose value requires NFC-normalized vertex
    labels; hypergraphs whose declared labels are not NFC are outside this
    operation's admitted domain.
    """

    hypergraph: FiniteHypergraph


class CliqueExpansionResult(StrictModel):
    """The primal/2-section graph of a finite hypergraph.

    ``graph`` is the canonical :class:`SimpleUndirectedGraph` whose vertices
    are exactly the hypergraph's declared vertex labels, in declared order,
    and in which two distinct vertices are adjacent if and only if they
    share at least one hyperedge.  Each edge's endpoints appear in lexical
    order per the canonical graph value's own convention, independent of the
    source hypergraph's declared ordering, so the value composes directly
    with downstream graph operations without translation.  This defining
    property is checked by the explicit owner verifier against the retained
    source hypergraph.
    """

    hypergraph: FiniteHypergraph
    graph: SimpleUndirectedGraph


# ---------------------------------------------------------------------------
# Induced uniform-hypergraph type profiles
# --------------------------------------------------------------------------

MAX_INDUCED_SUBSETS = 4_096
MAX_INDUCED_SUBSET_SIZE = MAX_VERTICES
MAX_INDUCED_PROFILE_RESULT_BYTES = CanonicalLimits().max_output_bytes


@dataclass(frozen=True, slots=True)
class _InducedTypeProfileAdmissionPlan:
    """Request-scoped work and output plan for an induced type profile."""

    expected_subsets: tuple[tuple[str, ...], ...]
    edge_sets: tuple[frozenset[str], ...]
    result_bytes: int


class InducedTypeProfileRequest(StrictModel):
    """Request the induced uniform type profile of a finite hypergraph.

    For each k-subset of the declared vertices, the operation reports the
    number of distinct nonempty induced edges ``e ∩ S`` that arise when the
    hypergraph is restricted to that subset ``S``.
    """

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Canonical finite hypergraph. Admission accounts for the exact "
            "UTF-8 label encoding of every returned subset and the retained "
            "source, in addition to the subset-count bound."
        ),
        json_schema_extra={
            "subset_count_bound": MAX_INDUCED_SUBSETS,
            "canonical_result_bytes_bound": MAX_INDUCED_PROFILE_RESULT_BYTES,
        },
    )
    subset_size: StrictInt = Field(
        ge=0,
        le=MAX_INDUCED_SUBSET_SIZE,
        description=(
            "Cardinality ``k`` of each inspected vertex subset.  The "
            f"complete profile admits at most {MAX_INDUCED_SUBSETS} "
            "k-subsets."
        ),
    )


def _strict_json_array_size(item_sizes: tuple[int, ...]) -> int:
    """Return the exact canonical JSON size of an array from item sizes."""

    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _induced_profile_result_bytes(
    hypergraph: FiniteHypergraph,
    expected_subsets: tuple[tuple[str, ...], ...],
) -> int:
    """Return a safe serialized-size bound for an induced-profile result.

    The count for each entry is not known until the kernel computes it, so its
    widest possible value is reserved. Subset labels and the retained source
    are measured using the same canonical normalization as the transport
    boundary, rather than a fixed per-entry estimate.
    """

    subset_size = len(expected_subsets[0])
    vertex_wire_bytes = {
        vertex: len(encode_strict_json(unicodedata.normalize("NFC", vertex)))
        for vertex in hypergraph.vertices
    }
    maximum_count = min(len(hypergraph.edges), (1 << subset_size) - 1)
    count_wire_bytes = len(encode_strict_json(maximum_count))
    entry_sizes: list[int] = []
    for subset in expected_subsets:
        subset_wire_bytes = _strict_json_array_size(
            tuple(vertex_wire_bytes[vertex] for vertex in subset)
        )
        entry_sizes.append(
            strict_json_object_size(
                (
                    ("vertex_subset", subset_wire_bytes),
                    ("induced_edge_count", count_wire_bytes),
                )
            )
        )
    entries_size = _strict_json_array_size(tuple(entry_sizes))
    source_size = len(canonicalize_json(hypergraph.model_dump(mode="json")))
    return strict_json_object_size(
        (
            ("hypergraph", source_size),
            ("subset_size", len(encode_strict_json(subset_size))),
            ("entries", entries_size),
        )
    )


def _induced_type_profile_admission_plan(
    hypergraph: FiniteHypergraph,
    subset_size: int,
) -> _InducedTypeProfileAdmissionPlan:
    """Build the complete bounded-work plan once for one profile request."""

    from itertools import combinations

    n = len(hypergraph.vertices)
    if subset_size > n:
        raise _validation_error(
            "induced type profile subset_size exceeds the declared vertex count"
        )
    expected_subsets = tuple(
        tuple(combo) for combo in combinations(sorted(hypergraph.vertices), subset_size)
    )
    if len(expected_subsets) > MAX_INDUCED_SUBSETS:
        raise _validation_error(
            "induced type profile exceeds the "
            f"{MAX_INDUCED_SUBSETS}-subset profile bound"
        )
    return _InducedTypeProfileAdmissionPlan(
        expected_subsets=expected_subsets,
        edge_sets=tuple(frozenset(members) for _, members in hypergraph.edges),
        result_bytes=_induced_profile_result_bytes(hypergraph, expected_subsets),
    )


class InducedTypeProfileEntry(StrictModel):
    """One vertex subset and its induced distinct-edge count."""

    vertex_subset: tuple[str, ...] = Field(max_length=MAX_INDUCED_SUBSET_SIZE)
    induced_edge_count: int = Field(ge=0, le=MAX_EDGES)

    @model_validator(mode="after")
    def require_canonical_entry(self) -> Self:
        if self.vertex_subset != tuple(sorted(set(self.vertex_subset))):
            raise _validation_error(
                "induced type profile vertex_subset must be distinct and "
                "lexicographically sorted"
            )
        return self


class InducedTypeProfileResult(StrictModel):
    """The complete induced uniform type profile of a finite hypergraph.

    ``entries`` lists one :class:`InducedTypeProfileEntry` per k-subset of the
    declared vertices, in lexicographic vertex order.  ``induced_edge_count``
    for a subset ``S`` is the number of distinct nonempty edges ``e ∩ S``
    arising from the source hypergraph's edges.
    """

    hypergraph: FiniteHypergraph
    subset_size: StrictInt = Field(ge=0, le=MAX_INDUCED_SUBSET_SIZE)
    entries: tuple[InducedTypeProfileEntry, ...] = Field(max_length=MAX_INDUCED_SUBSETS)

    @classmethod
    def _from_kernel(
        cls,
        *,
        hypergraph: FiniteHypergraph,
        subset_size: int,
        entries: tuple[InducedTypeProfileEntry, ...],
    ) -> Self:
        """Construct output after the admitted owner-local kernel completes."""

        return cls.model_construct(
            hypergraph=hypergraph,
            subset_size=subset_size,
            entries=entries,
        )


# ---------------------------------------------------------------------------
# Bounded exact minimum vertex transversals
# --------------------------------------------------------------------------

MAX_TRANSVERSAL_RESULT_VERTICES = MAX_VERTICES
MAX_TRANSVERSAL_SEARCH_WORK = 50_000_000
MAX_TRANSVERSAL_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _minimum_transversal_search_plan(
    hypergraph: FiniteHypergraph,
) -> tuple[tuple[str, ...], tuple[frozenset[str], ...], int, int]:
    """Return active vertices, unique edges, search depth, and work bound."""

    from math import comb

    unique_edges: list[frozenset[str]] = []
    seen_edges: set[frozenset[str]] = set()
    for _, members in hypergraph.edges:
        edge = frozenset(members)
        if edge not in seen_edges:
            seen_edges.add(edge)
            unique_edges.append(edge)
    unique_edges.sort(key=lambda edge: (len(edge), tuple(sorted(edge))))
    active_labels = set().union(*unique_edges) if unique_edges else set()
    active_vertices = tuple(
        vertex for vertex in hypergraph.vertices if vertex in active_labels
    )

    greedy_witness: set[str] = set()
    for edge in unique_edges:
        if not greedy_witness & edge:
            greedy_witness.add(
                next(vertex for vertex in active_vertices if vertex in edge)
            )
    search_depth = len(greedy_witness)
    candidate_count = sum(
        comb(len(active_vertices), size) for size in range(1, search_depth + 1)
    )
    return (
        active_vertices,
        tuple(unique_edges),
        search_depth,
        candidate_count * len(unique_edges),
    )


def _minimum_transversal_result_bytes(
    hypergraph: FiniteHypergraph,
    active_vertices: tuple[str, ...],
) -> int:
    """Return the exact worst-case canonical size of a transversal result."""

    normalized_active_vertices = tuple(
        unicodedata.normalize("NFC", vertex) for vertex in active_vertices
    )
    source_size = len(canonicalize_json(hypergraph.model_dump(mode="json")))
    witness_size = _strict_json_array_size(
        tuple(_strict_label_wire_bytes(vertex) for vertex in normalized_active_vertices)
    )
    cardinality_size = len(encode_strict_json(len(normalized_active_vertices)))
    return strict_json_object_size(
        (
            ("hypergraph", source_size),
            ("transversal", witness_size),
            ("cardinality", cardinality_size),
        )
    )


class MinimumTransversalRequest(StrictModel):
    """Request an exact minimum-cardinality transversal (hitting set).

    A transversal is a set of vertices that intersects every hyperedge.
    The exact bounded search ignores carrier vertices absent from every edge,
    deduplicates the edge family, and enumerates active-vertex subsets by
    increasing cardinality. Admission bounds the resulting candidate/edge
    intersection checks by MAX_TRANSVERSAL_SEARCH_WORK.
    """

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Canonical finite hypergraph. Every hyperedge must be nonempty. "
            "Empty edge families are solved by the empty transversal; "
            "otherwise the exact active-vertex search must fit "
            f"{MAX_TRANSVERSAL_SEARCH_WORK} candidate-edge checks and the "
            f"retained-source result must fit {MAX_TRANSVERSAL_RESULT_BYTES} "
            "canonical bytes."
        ),
        json_schema_extra={
            "search_work_bound": MAX_TRANSVERSAL_SEARCH_WORK,
            "canonical_result_bytes_bound": MAX_TRANSVERSAL_RESULT_BYTES,
            "requires_nonempty_hyperedges": True,
        },
    )


class MinimumTransversalResult(StrictModel):
    """An exact minimum-cardinality transversal of a finite hypergraph.

    ``transversal`` is one minimum-cardinality vertex set intersecting every
    hyperedge, in declared vertex order.  ``cardinality`` is its size.  For a
    hypergraph with no edges, the empty set is the unique minimum transversal.
    """

    hypergraph: FiniteHypergraph
    transversal: tuple[str, ...] = Field(max_length=MAX_TRANSVERSAL_RESULT_VERTICES)
    cardinality: StrictInt = Field(ge=0, le=MAX_TRANSVERSAL_RESULT_VERTICES)

    @model_validator(mode="after")
    def bind_transversal(self) -> Self:
        vertex_set = set(self.hypergraph.vertices)
        witness_set = set(self.transversal)
        if len(witness_set) != len(self.transversal):
            raise _validation_error("transversal vertices must be distinct")
        if not witness_set <= vertex_set:
            raise _validation_error(
                "transversal vertices must be declared source vertices"
            )
        expected_order = tuple(
            vertex for vertex in self.hypergraph.vertices if vertex in witness_set
        )
        if self.transversal != expected_order:
            raise _validation_error(
                "transversal vertices must be unique and in declared vertex order"
            )
        if self.cardinality != len(self.transversal):
            raise _validation_error("cardinality must equal the transversal size")
        return self


# ---------------------------------------------------------------------------
# Bounded exact maximum edge-matchings
# --------------------------------------------------------------------------

MAX_MATCHING_EDGES = 20
MAX_MATCHING_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _maximum_edge_matching_result_bytes(
    hypergraph: FiniteHypergraph,
    edge_ids: tuple[str, ...],
) -> int:
    """Return the exact canonical size of a worst-case matching result."""

    normalized_edge_ids = tuple(
        unicodedata.normalize("NFC", edge_id) for edge_id in edge_ids
    )
    source_size = len(canonicalize_json(hypergraph.model_dump(mode="json")))
    matching_size = _strict_json_array_size(
        tuple(_strict_label_wire_bytes(edge_id) for edge_id in normalized_edge_ids)
    )
    count_size = len(encode_strict_json(len(normalized_edge_ids)))
    return strict_json_object_size(
        (
            ("hypergraph", source_size),
            ("matching", matching_size),
            ("count", count_size),
        )
    )


class MaximumEdgeMatchingRequest(StrictModel):
    """Request an exact maximum-cardinality edge matching.

    A matching is a set of pairwise-disjoint hyperedges.  The exact bounded
    search enumerates nonempty edge subsets by decreasing cardinality and
    admits at most ``MAX_MATCHING_EDGES`` search edges.  Empty edges form a
    mandatory witness prefix because they are disjoint from every edge,
    subject to the canonical result-byte bound.
    """

    hypergraph: FiniteHypergraph = Field(
        description=(
            "Canonical finite hypergraph. Exact matching search admits at most "
            f"{MAX_MATCHING_EDGES} nonempty edges; empty edge IDs are included "
            "in every matching witness and the result is admitted when its "
            f"retained-source result fits {MAX_MATCHING_RESULT_BYTES} canonical bytes."
        ),
        json_schema_extra={
            "search_edge_bound": MAX_MATCHING_EDGES,
            "canonical_result_bytes_bound": MAX_MATCHING_RESULT_BYTES,
            "empty_edge_witness_prefix": True,
        },
    )


class MaximumEdgeMatchingResult(StrictModel):
    """An exact maximum-cardinality edge matching of a finite hypergraph.

    ``matching`` is one maximum-cardinality set of pairwise-disjoint
    hyperedge ids, in declared edge order.  ``count`` is its size.  For a
    hypergraph with no edges, the empty set is the unique maximum matching.
    """

    hypergraph: FiniteHypergraph
    matching: tuple[str, ...] = Field(max_length=MAX_EDGES)
    count: StrictInt = Field(ge=0, le=MAX_EDGES)

    @model_validator(mode="after")
    def bind_matching(self) -> Self:
        edge_ids = tuple(edge_id for edge_id, _ in self.hypergraph.edges)
        edge_id_set = set(edge_ids)
        witness_set = set(self.matching)
        if len(witness_set) != len(self.matching):
            raise _validation_error("matching edge ids must be distinct")
        if not witness_set <= edge_id_set:
            raise _validation_error(
                "matching edge ids must be declared source edge ids"
            )
        expected_order = tuple(
            edge_id for edge_id in edge_ids if edge_id in witness_set
        )
        if self.matching != expected_order:
            raise _validation_error(
                "matching edge ids must be unique and in declared edge order"
            )
        if self.count != len(self.matching):
            raise _validation_error("count must equal the matching size")
        return self
