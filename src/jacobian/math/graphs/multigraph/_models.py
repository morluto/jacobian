"""Typed wire contracts for finite multigraph flow and cycle operations.

All operations in this module act on a finite loopless multigraph supplied
with explicit edge IDs so that parallel edges are never conflated.  A finite
Abelian group is represented concretely as a product of cyclic groups with
bounded positive moduli and canonical residue coordinates.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, StrictStr, model_validator

from jacobian._models import StrictModel

__all__ = [
    "CycleMulticoverRequest",
    "CycleMulticoverResult",
    "CycleRecord",
    "EulerianCyclesRequest",
    "EulerianCyclesResult",
    "FiniteAbelianGroup",
    "FlowEdgeAssignment",
    "GroupElement",
    "LooplessMultigraph",
    "MultigraphEdge",
    "MultigraphFlowCheckRequest",
    "MultigraphFlowCheckResult",
    "MultigraphFlowFindRequest",
    "MultigraphFlowFindResult",
    "MultigraphFlowSearchBudget",
    "VertexDivergence",
]

# Public bounds
MAX_VERTICES = 32
MAX_EDGES = 128
MAX_PARALLEL_MULTIPLICITY = 32
MAX_GROUP_RANK = 6
MAX_GROUP_MODULUS = 4096
MAX_GROUP_CARDINALITY = 4096
MAX_FLOW_SEARCH_STATES = 1_048_576
MAX_CYCLE_COUNT = 64
MAX_CYCLE_LENGTH = 64
MAX_CYCLE_EDGE_INCIDENCES = 4096


# ---------------------------------------------------------------------------
# Finite loopless multigraph
# ---------------------------------------------------------------------------


class MultigraphEdge(StrictModel):
    """One edge of a loopless multigraph with explicit identity.

    ``edge_id`` carries identity; parallel edges sharing the same endpoint
    pair are allowed because each has a distinct ``edge_id``.  Endpoints are
    zero-based vertex indices; ``left`` and ``right`` must be distinct (no
    loops) and need not be ordered.
    """

    edge_id: StrictStr = Field(min_length=1, max_length=64)
    left: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    right: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)

    @model_validator(mode="after")
    def require_loopless(self) -> Self:
        if self.left == self.right:
            raise ValueError("multigraph edges must be loopless (distinct endpoints)")
        return self


class LooplessMultigraph(StrictModel):
    """Immutable canonical value for a finite loopless multigraph.

    Vertices are ``0..vertex_count-1``.  Edge IDs are unique; parallel edges
    are distinguished by their IDs, not by endpoint tuples.  Edge order is
    deterministic (input order is preserved) but IDs carry identity.
    """

    vertex_count: StrictInt = Field(ge=0, le=MAX_VERTICES)
    edges: tuple[MultigraphEdge, ...] = Field(max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_canonical_multigraph(self) -> Self:
        seen_ids: set[str] = set()
        for edge in self.edges:
            if edge.edge_id in seen_ids:
                raise ValueError("multigraph edge IDs must be unique")
            seen_ids.add(edge.edge_id)
            if not (
                0 <= edge.left < self.vertex_count
                and 0 <= edge.right < self.vertex_count
            ):
                raise ValueError("edge endpoints must be in 0..vertex_count-1")
        return self

    @property
    def vertex_set(self) -> frozenset[int]:
        return frozenset(range(self.vertex_count))

    @property
    def edge_id_set(self) -> frozenset[str]:
        return frozenset(edge.edge_id for edge in self.edges)

    def edge_by_id(self, edge_id: str) -> MultigraphEdge:
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge
        raise KeyError(edge_id)


# ---------------------------------------------------------------------------
# Finite Abelian group (product of cyclic groups)
# ---------------------------------------------------------------------------


class FiniteAbelianGroup(StrictModel):
    """A finite Abelian group represented as a product of cyclic groups.

    The group is ``(Z/n1Z) x ... x (Z/nrZ)`` with bounded positive moduli.
    Elements are canonical residue tuples ``(0 <= x_i < n_i)``.  Addition and
    negation are componentwise modulo the respective moduli.  The zero element
    is ``(0, ..., 0)``.
    """

    moduli: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_GROUP_RANK)

    @model_validator(mode="after")
    def require_valid_moduli(self) -> Self:
        product = 1
        for modulus in self.moduli:
            if modulus < 2:
                raise ValueError("group moduli must be at least 2")
            if modulus > MAX_GROUP_MODULUS:
                raise ValueError(f"group modulus exceeds {MAX_GROUP_MODULUS}")
            product *= modulus
            if product > MAX_GROUP_CARDINALITY:
                raise ValueError(f"group cardinality exceeds {MAX_GROUP_CARDINALITY}")
        return self

    @property
    def rank(self) -> int:
        return len(self.moduli)

    @property
    def cardinality(self) -> int:
        product = 1
        for modulus in self.moduli:
            product *= modulus
        return product

    @property
    def zero(self) -> tuple[int, ...]:
        return tuple(0 for _ in self.moduli)

    def add(self, left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        if len(left) != self.rank or len(right) != self.rank:
            raise ValueError("element rank must match group rank")
        return tuple(
            (a + b) % m for a, b, m in zip(left, right, self.moduli, strict=True)
        )

    def negate(self, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != self.rank:
            raise ValueError("element rank must match group rank")
        return tuple((-a) % m for a, m in zip(value, self.moduli, strict=True))

    def sum(self, values: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Sum a tuple of group elements; empty sum is zero."""
        result = self.zero
        for value in values:
            result = self.add(result, value)
        return result

    def normalize(self, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reduce each coordinate into the canonical range ``0 <= x_i < n_i``."""
        if len(value) != self.rank:
            raise ValueError("element rank must match group rank")
        return tuple(a % m for a, m in zip(value, self.moduli, strict=True))

    def is_zero(self, value: tuple[int, ...]) -> bool:
        normalized = self.normalize(value)
        return all(a == 0 for a in normalized)


class GroupElement(StrictModel):
    """A canonical residue-tuple element of a finite Abelian group."""

    coordinates: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_GROUP_RANK)

    @model_validator(mode="after")
    def require_non_negative(self) -> Self:
        for coordinate in self.coordinates:
            if coordinate < 0:
                raise ValueError("group element coordinates must be non-negative")
        return self


# ---------------------------------------------------------------------------
# Oriented flow records
# ---------------------------------------------------------------------------


class FlowEdgeAssignment(StrictModel):
    """One oriented edge-value binding for a flow.

    ``edge_id`` references a multigraph edge.  ``orientation`` is the tail-to-
    head direction: when ``orientation`` equals ``left_to_right`` the tail is
    ``edge.left`` and the head is ``edge.right``; when ``right_to_left`` the
    tail is ``edge.right`` and the head is ``edge.left``.  ``value`` is the
    group element assigned at the tail.  Reversing the orientation must be
    paired with negating the value to represent the same flow.
    """

    edge_id: StrictStr = Field(min_length=1, max_length=64)
    orientation: Literal["left_to_right", "right_to_left"]
    value: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_GROUP_RANK)

    @model_validator(mode="after")
    def require_non_negative_value(self) -> Self:
        for coordinate in self.value:
            if coordinate < 0:
                raise ValueError("flow values must be non-negative residues")
        return self


class VertexDivergence(StrictModel):
    """Per-vertex signed divergence (net flow out minus in) for one vertex.

    ``coordinates`` is the group-element divergence at this vertex; the zero
    element means conservation holds here.  ``incident_edge_ids`` lists the
    edge IDs contributing to the signed sum, in deterministic order.
    """

    vertex: StrictInt = Field(ge=0, le=MAX_VERTICES - 1)
    coordinates: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_GROUP_RANK)
    incident_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    conservation_holds: bool


class MultigraphFlowCheckRequest(StrictModel):
    """Request to check a finite-Abelian flow on a loopless multigraph."""

    graph: LooplessMultigraph
    group: FiniteAbelianGroup
    edge_values: tuple[FlowEdgeAssignment, ...] = Field(max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_complete_assignment(self) -> Self:
        graph_ids = self.graph.edge_id_set
        assigned_ids = {assign.edge_id for assign in self.edge_values}
        if graph_ids != assigned_ids:
            missing = sorted(graph_ids - assigned_ids)
            extra = sorted(assigned_ids - graph_ids)
            raise ValueError(
                f"edge_values must assign every edge exactly once; "
                f"missing={missing}, extra={extra}"
            )
        if len(assigned_ids) != len(self.edge_values):
            raise ValueError("edge_values must not repeat edge IDs")
        for assign in self.edge_values:
            if len(assign.value) != self.group.rank:
                raise ValueError("flow value rank must match group rank")
            if any(
                c >= m for c, m in zip(assign.value, self.group.moduli, strict=True)
            ):
                raise ValueError("flow value coordinates must be below their moduli")
        return self


class MultigraphFlowCheckResult(StrictModel):
    """Exact per-vertex conservation ledger for a submitted flow.

    ``edge_flow_records`` are the canonical oriented edge-flow records.  The
    ``divergence_ledger`` gives the per-vertex signed divergence.  ``zero_edge_ids``
    lists edges assigned the zero group element.  ``nowhere_zero`` is true
    when no edge is zero.  ``conservation_holds`` is true when every vertex
    has zero divergence.
    """

    result_schema_version: Literal["1"] = "1"
    edge_flow_records: tuple[FlowEdgeAssignment, ...] = Field(max_length=MAX_EDGES)
    divergence_ledger: tuple[VertexDivergence, ...] = Field(max_length=MAX_VERTICES)
    zero_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    nowhere_zero: bool
    conservation_holds: bool


# ---------------------------------------------------------------------------
# Bounded flow search
# ---------------------------------------------------------------------------


class MultigraphFlowSearchBudget(StrictModel):
    """Explicit public limits for one bounded finite-Abelian flow search."""

    max_states: StrictInt = Field(
        default=MAX_FLOW_SEARCH_STATES,
        ge=1,
        le=MAX_FLOW_SEARCH_STATES,
        description=(
            "Maximum number of flow assignment states the search may visit "
            "before declaring UNKNOWN."
        ),
    )
    require_nowhere_zero: bool = True


class MultigraphFlowFindRequest(StrictModel):
    """Request to search for a finite-Abelian flow on a loopless multigraph.

    When ``require_nowhere_zero`` is true the search looks for a nowhere-zero
    flow; otherwise any flow (including zero-valued edges) suffices.  The
    search is bounded by ``resource_budget.max_states``; a complete
    exhaustive search returns ``EXHAUSTED`` when no flow exists in the
    declared finite domain, and ``UNKNOWN`` when the budget is exceeded
    before the search completes.
    """

    graph: LooplessMultigraph
    group: FiniteAbelianGroup
    resource_budget: MultigraphFlowSearchBudget = Field(
        default_factory=MultigraphFlowSearchBudget
    )


class MultigraphFlowFindResult(StrictModel):
    """Outcome of a bounded finite-Abelian flow search.

    ``status`` is one of ``FOUND``, ``EXHAUSTED``, ``UNKNOWN``.  When ``FOUND``,
    ``flow`` is the witness (checked before return).  When ``EXHAUSTED`` the
    complete declared finite search space was covered and no flow exists.
    When ``UNKNOWN`` the resource limit was hit before completion.
    """

    result_schema_version: Literal["1"] = "1"
    status: Literal["FOUND", "EXHAUSTED", "UNKNOWN"]
    flow: tuple[FlowEdgeAssignment, ...] | None = Field(
        default=None, max_length=MAX_EDGES
    )
    states_explored: StrictInt = Field(ge=0)
    termination_reason: Literal[
        "WITNESS_FOUND",
        "SEARCH_EXHAUSTED",
        "STATE_BUDGET_EXCEEDED",
        "SPECIAL_CASE",
    ]

    @model_validator(mode="after")
    def require_consistent_status(self) -> Self:
        if self.status == "FOUND":
            if self.flow is None:
                raise ValueError("FOUND status requires a flow witness")
            if self.termination_reason not in ("WITNESS_FOUND", "SPECIAL_CASE"):
                raise ValueError(
                    "FOUND status requires WITNESS_FOUND or SPECIAL_CASE reason"
                )
        elif self.status == "EXHAUSTED":
            if self.flow is not None:
                raise ValueError("EXHAUSTED status must not include a flow")
            if self.termination_reason != "SEARCH_EXHAUSTED":
                raise ValueError("EXHAUSTED status requires SEARCH_EXHAUSTED reason")
        elif self.status == "UNKNOWN":
            if self.flow is not None:
                raise ValueError("UNKNOWN status must not include a flow")
            if self.termination_reason != "STATE_BUDGET_EXCEEDED":
                raise ValueError("UNKNOWN status requires STATE_BUDGET_EXCEEDED")
        return self


# ---------------------------------------------------------------------------
# Eulerian cycle decomposition
# ---------------------------------------------------------------------------


class EulerianCyclesRequest(StrictModel):
    """Request to decompose an even-parity edge multiset into cycles.

    When ``edge_subset`` is omitted, the full edge set of ``graph`` is used.
    When provided, it must reference existing edge IDs and the induced degree
    of every vertex must be even (Eulerian condition).  The result is one
    deterministic decomposition into edge-disjoint cycles.
    """

    graph: LooplessMultigraph
    edge_subset: tuple[StrictStr, ...] | None = Field(
        default=None, max_length=MAX_EDGES
    )

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        if self.edge_subset is not None:
            graph_ids = self.graph.edge_id_set
            subset_set = set(self.edge_subset)
            if not subset_set.issubset(graph_ids):
                missing = sorted(subset_set - graph_ids)
                raise ValueError(f"edge_subset references unknown edge IDs: {missing}")
            if len(subset_set) != len(self.edge_subset):
                raise ValueError("edge_subset must not repeat edge IDs")
        return self


class CycleRecord(StrictModel):
    """One explicit cycle as a closed alternating vertex/edge-ID sequence.

    ``vertices`` and ``edge_ids`` alternate so that ``edge_ids[i]`` connects
    ``vertices[i]`` and ``vertices[(i+1) % len]``.  The sequence is closed:
    the first vertex equals the last, and ``len(vertices) == len(edge_ids) + 1``.
    """

    vertices: tuple[StrictInt, ...] = Field(min_length=3, max_length=MAX_CYCLE_LENGTH)
    edge_ids: tuple[StrictStr, ...] = Field(min_length=2, max_length=MAX_CYCLE_LENGTH)

    @model_validator(mode="after")
    def require_closed_cycle(self) -> Self:
        if len(self.vertices) != len(self.edge_ids) + 1:
            raise ValueError("a cycle must have len(vertices) == len(edge_ids) + 1")
        if self.vertices[0] != self.vertices[-1]:
            raise ValueError("a cycle must be closed (first vertex == last)")
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("a cycle must not repeat edge IDs")
        return self


class EulerianCyclesResult(StrictModel):
    """Deterministic edge-disjoint cycle decomposition and edge-usage profile.

    ``cycles`` are edge-disjoint and together cover every edge in the
    requested edge subset exactly once.  ``edge_usage`` maps each covered
    edge ID to its multiplicity (always 1 for a decomposition).  ``covers_all``
    is true when every requested edge appears in some cycle.
    """

    result_schema_version: Literal["1"] = "1"
    cycles: tuple[CycleRecord, ...] = Field(max_length=MAX_CYCLE_COUNT)
    edge_usage: tuple[tuple[StrictStr, StrictInt], ...] = Field(max_length=MAX_EDGES)
    covers_all: bool


# ---------------------------------------------------------------------------
# Cycle multicover check
# ---------------------------------------------------------------------------


class CycleMulticoverRequest(StrictModel):
    """Request to check that a cycle family covers each edge exactly k times.

    The ``cycles`` are validated against ``graph`` incidence.  Each cycle may
    appear in any ordering, rotation, or reversal; the operation scores edge
    multiplicity, not one rendering.  ``target_multiplicity`` is the required
    per-edge multiplicity ``k``.
    """

    graph: LooplessMultigraph
    cycles: tuple[CycleRecord, ...] = Field(max_length=MAX_CYCLE_COUNT)
    target_multiplicity: StrictInt = Field(ge=0, le=MAX_PARALLEL_MULTIPLICITY)

    @model_validator(mode="after")
    def require_bounded_incidence(self) -> Self:
        total = sum(len(cycle.edge_ids) for cycle in self.cycles)
        if total > MAX_CYCLE_EDGE_INCIDENCES:
            raise ValueError(
                f"total cycle-edge incidences exceed {MAX_CYCLE_EDGE_INCIDENCES}"
            )
        return self


class CycleMulticoverResult(StrictModel):
    """Per-cycle validity and exact edge-multiplicity profile for a cover.

    ``cycle_validity`` is a per-row boolean: true when the cycle follows graph
    incidence (each consecutive edge connects consecutive vertices, and every
    edge ID exists in the graph).  ``edge_multiplicity`` maps each graph edge
    ID to the number of times the cycle family traverses it.  ``missing_edge_ids``
    are edges covered fewer than ``k`` times; ``overcovered_edge_ids`` are
    edges covered more than ``k`` times.  ``is_exact_k_cover`` is true when
    every edge is covered exactly ``k`` times and all cycles are valid.
    """

    result_schema_version: Literal["1"] = "1"
    cycle_validity: tuple[bool, ...] = Field(max_length=MAX_CYCLE_COUNT)
    edge_multiplicity: tuple[tuple[StrictStr, StrictInt], ...] = Field(
        max_length=MAX_EDGES
    )
    missing_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    overcovered_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    is_exact_k_cover: bool
