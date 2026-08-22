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
    """Exact per-vertex conservation ledger for a submitted flow, bound to its source.

    ``graph`` and ``group`` bind the conclusion to the checked request so the
    ledger, zero-edge set, and booleans remain reconstructible.
    """

    result_schema_version: Literal["1"] = "1"
    graph: LooplessMultigraph
    group: FiniteAbelianGroup
    edge_flow_records: tuple[FlowEdgeAssignment, ...] = Field(max_length=MAX_EDGES)
    divergence_ledger: tuple[VertexDivergence, ...] = Field(max_length=MAX_VERTICES)
    zero_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    nowhere_zero: bool
    conservation_holds: bool

    @model_validator(mode="after")
    def require_bound_flow_check(self) -> Self:
        # Validate that edge_flow_records cover exactly graph edges
        graph_ids = self.graph.edge_id_set
        record_ids = {r.edge_id for r in self.edge_flow_records}
        if graph_ids != record_ids:
            raise ValueError("edge_flow_records must cover exactly graph edges")
        if len(record_ids) != len(self.edge_flow_records):
            raise ValueError("edge_flow_records must not repeat edge IDs")
        for rec in self.edge_flow_records:
            if len(rec.value) != self.group.rank:
                raise ValueError("flow value rank must match group rank")
            if any(c >= m for c, m in zip(rec.value, self.group.moduli, strict=True)):
                raise ValueError("flow value coordinates must be below moduli")
            if any(c < 0 for c in rec.value):
                raise ValueError("flow values must be non-negative")
        # Recompute divergence ledger
        vertex_out: dict[int, list[tuple[int, ...]]] = {
            v: [] for v in range(self.graph.vertex_count)
        }
        vertex_in: dict[int, list[tuple[int, ...]]] = {
            v: [] for v in range(self.graph.vertex_count)
        }
        vertex_incident: dict[int, set[str]] = {
            v: set() for v in range(self.graph.vertex_count)
        }
        for rec in self.edge_flow_records:
            edge = self.graph.edge_by_id(rec.edge_id)
            tail, head = (edge.left, edge.right) if rec.orientation == "left_to_right" else (edge.right, edge.left)
            val = self.group.normalize(rec.value)
            vertex_out[tail].append(val)
            vertex_in[head].append(val)
            vertex_incident[tail].add(rec.edge_id)
            vertex_incident[head].add(rec.edge_id)
        expected_ledger: list[VertexDivergence] = []
        expected_conservation = True
        for v in range(self.graph.vertex_count):
            out_sum = self.group.sum(tuple(vertex_out[v]))
            in_sum = self.group.sum(tuple(vertex_in[v]))
            div = self.group.add(out_sum, self.group.negate(in_sum))
            holds = self.group.is_zero(div)
            if not holds:
                expected_conservation = False
            expected_ledger.append(
                VertexDivergence(
                    vertex=v,
                    coordinates=div,
                    incident_edge_ids=tuple(sorted(vertex_incident[v])),
                    conservation_holds=holds,
                )
            )
        if tuple(expected_ledger) != self.divergence_ledger:
            raise ValueError("divergence_ledger does not match recomputed ledger")
        if self.conservation_holds != expected_conservation:
            raise ValueError("conservation_holds does not match recomputed value")
        expected_zero = tuple(sorted(rec.edge_id for rec in self.edge_flow_records if self.group.is_zero(rec.value)))
        if self.zero_edge_ids != expected_zero:
            raise ValueError("zero_edge_ids does not match recomputed zero set")
        if self.nowhere_zero != (len(expected_zero) == 0):
            raise ValueError("nowhere_zero does not match zero_edge set")
        return self


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
    """Outcome of a bounded finite-Abelian flow search, bound to its source request."""

    result_schema_version: Literal["1"] = "1"
    graph: LooplessMultigraph | None = None
    group: FiniteAbelianGroup | None = None
    resource_budget: MultigraphFlowSearchBudget | None = None
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
        # If source is bound, validate budget and witness
        if self.graph is not None and self.group is not None and self.resource_budget is not None:
            if self.states_explored > self.resource_budget.max_states:
                raise ValueError("states_explored exceeds resource budget")
            if self.status == "FOUND" and self.flow is not None:
                # Validate flow witness is a valid flow for graph/group
                from jacobian.math.graphs.multigraph._models import MultigraphFlowCheckRequest

                # Check that flow covers all edges and respects group
                req = MultigraphFlowCheckRequest(
                    graph=self.graph, group=self.group, edge_values=self.flow
                )
                # Check conservation and nowhere_zero if required
                # Recompute via check logic without recursion
                graph_ids = self.graph.edge_id_set
                assigned = {a.edge_id for a in self.flow}
                if graph_ids != assigned:
                    raise ValueError("FOUND flow must assign every graph edge")
                # Check budget's nowhere_zero requirement if present
                if self.resource_budget.require_nowhere_zero:
                    for a in self.flow:
                        if self.group.is_zero(a.value):
                            raise ValueError("FOUND flow violates nowhere_zero requirement")
                # Conservation check via group
                vertex_out: dict[int, list[tuple[int, ...]]] = {v: [] for v in range(self.graph.vertex_count)}
                vertex_in: dict[int, list[tuple[int, ...]]] = {v: [] for v in range(self.graph.vertex_count)}
                for a in self.flow:
                    edge = self.graph.edge_by_id(a.edge_id)
                    tail, head = (edge.left, edge.right) if a.orientation == "left_to_right" else (edge.right, edge.left)
                    val = self.group.normalize(a.value)
                    vertex_out[tail].append(val)
                    vertex_in[head].append(val)
                for v in range(self.graph.vertex_count):
                    out_sum = self.group.sum(tuple(vertex_out[v]))
                    in_sum = self.group.sum(tuple(vertex_in[v]))
                    if not self.group.is_zero(self.group.add(out_sum, self.group.negate(in_sum))):
                        raise ValueError("FOUND flow does not satisfy conservation")
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
        if any(v < 0 or v >= MAX_VERTICES for v in self.vertices):
            raise ValueError(f"cycle vertices must be in 0..{MAX_VERTICES - 1}")
        if len(set(self.vertices[:-1])) != len(self.vertices) - 1:
            raise ValueError(
                "a cycle must not repeat interior vertices "
                "(all vertices except the closing duplicate must be distinct)"
            )
        return self


class EulerianCyclesResult(StrictModel):
    """Deterministic edge-disjoint cycle decomposition, bound to its source edge set."""

    result_schema_version: Literal["1"] = "1"
    graph: LooplessMultigraph | None = None
    edge_subset: tuple[StrictStr, ...] | None = Field(default=None, max_length=MAX_EDGES)
    cycles: tuple[CycleRecord, ...] = Field(max_length=MAX_CYCLE_COUNT)
    edge_usage: tuple[tuple[StrictStr, StrictInt], ...] = Field(max_length=MAX_EDGES)
    covers_all: bool

    @model_validator(mode="after")
    def require_bound_eulerian(self) -> Self:
        if self.graph is None:
            return self
        # Determine requested edge IDs
        if self.edge_subset is not None:
            requested = set(self.edge_subset)
            if not requested.issubset(self.graph.edge_id_set):
                raise ValueError("edge_subset contains unknown edge IDs")
        else:
            requested = set(self.graph.edge_id_set)
        # Validate each cycle is valid and edge-disjoint via CycleRecord validators
        # already ensure cycles are closed and interior vertices distinct
        used: dict[str, int] = {eid: 0 for eid in requested}
        seen_cycle_ids: set[str] = set()
        for cycle in self.cycles:
            for eid in cycle.edge_ids:
                if eid not in requested:
                    raise ValueError(f"cycle uses edge {eid} not in requested set")
                if eid in seen_cycle_ids:
                    raise ValueError(f"edge {eid} appears in multiple cycles (must be disjoint)")
                seen_cycle_ids.add(eid)
                # Validate incidence
                idx = cycle.edge_ids.index(eid)  # not needed, but check per cycle
                # Find position of eid in this cycle
                pos = list(cycle.edge_ids).index(eid)
                v_from = cycle.vertices[pos]
                v_to = cycle.vertices[pos + 1]
                edge = self.graph.edge_by_id(eid)
                if not ((edge.left == v_from and edge.right == v_to) or (edge.right == v_from and edge.left == v_to)):
                    raise ValueError(f"cycle edge {eid} incidence does not match graph")
                used[eid] += 1
        # Validate edge_usage matches
        expected_usage = tuple(sorted((eid, used[eid]) for eid in sorted(requested)))
        # edge_usage is expected to be sorted tuple
        if self.edge_usage != expected_usage:
            raise ValueError(f"edge_usage {self.edge_usage} does not match recomputed {expected_usage}")
        expected_covers = all(used[eid] == 1 for eid in requested) and len(requested) > 0 or (len(requested) == 0 and len(self.cycles) == 0)
        # For empty requested, covers_all should be True per operation
        if len(requested) == 0:
            expected_covers = True
        if self.covers_all != expected_covers:
            raise ValueError(f"covers_all {self.covers_all} does not match expected {expected_covers}")
        return self


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

    The result retains its source (``graph``, ``cycles``, ``target_multiplicity``)
    so the exact-cover conclusion is reconstructible and bound to the submitted
    request.
    """

    result_schema_version: Literal["1"] = "1"
    graph: LooplessMultigraph
    cycles: tuple[CycleRecord, ...] = Field(max_length=MAX_CYCLE_COUNT)
    target_multiplicity: StrictInt = Field(ge=0, le=MAX_PARALLEL_MULTIPLICITY)
    cycle_validity: tuple[bool, ...] = Field(max_length=MAX_CYCLE_COUNT)
    edge_multiplicity: tuple[tuple[StrictStr, StrictInt], ...] = Field(
        max_length=MAX_EDGES
    )
    missing_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    overcovered_edge_ids: tuple[StrictStr, ...] = Field(max_length=MAX_EDGES)
    is_exact_k_cover: bool

    @model_validator(mode="after")
    def require_consistent_cover(self) -> Self:  # noqa: C901
        if len(self.cycle_validity) != len(self.cycles):
            raise ValueError("cycle_validity length must match cycles length")
        # Recompute validity and multiplicity from the retained source.
        recomputed_multiplicity: dict[str, int] = {
            edge.edge_id: 0 for edge in self.graph.edges
        }
        recomputed_validity: list[bool] = []
        for cycle in self.cycles:
            cycle_valid = True
            for i, eid in enumerate(cycle.edge_ids):
                edge_valid = True
                if eid not in recomputed_multiplicity:
                    edge_valid = False
                    cycle_valid = False
                else:
                    v_from = cycle.vertices[i]
                    v_to = cycle.vertices[i + 1]
                    if not (
                        0 <= v_from < self.graph.vertex_count
                        and 0 <= v_to < self.graph.vertex_count
                    ):
                        edge_valid = False
                        cycle_valid = False
                    else:
                        edge = self.graph.edge_by_id(eid)
                        if not (
                            (edge.left == v_from and edge.right == v_to)
                            or (edge.right == v_from and edge.left == v_to)
                        ):
                            edge_valid = False
                            cycle_valid = False
                if edge_valid:
                    recomputed_multiplicity[eid] += 1
            recomputed_validity.append(cycle_valid)
        if tuple(recomputed_validity) != self.cycle_validity:
            raise ValueError(
                f"cycle_validity {self.cycle_validity} does not match "
                f"recomputed {tuple(recomputed_validity)} from source"
            )
        recomputed_tuple = tuple(
            (eid, recomputed_multiplicity[eid])
            for eid in sorted(recomputed_multiplicity)
        )
        if recomputed_tuple != self.edge_multiplicity:
            raise ValueError(
                f"edge_multiplicity {self.edge_multiplicity} does not match "
                f"recomputed {recomputed_tuple} from source"
            )
        k = self.target_multiplicity
        recomputed_missing = tuple(
            sorted(eid for eid, cnt in recomputed_multiplicity.items() if cnt < k)
        )
        recomputed_over = tuple(
            sorted(eid for eid, cnt in recomputed_multiplicity.items() if cnt > k)
        )
        if recomputed_missing != self.missing_edge_ids:
            raise ValueError(
                f"missing_edge_ids {self.missing_edge_ids} does not match "
                f"recomputed {recomputed_missing} for k={k}"
            )
        if recomputed_over != self.overcovered_edge_ids:
            raise ValueError(
                f"overcovered_edge_ids {self.overcovered_edge_ids} does not match "
                f"recomputed {recomputed_over} for k={k}"
            )
        expected_exact = (
            all(recomputed_validity)
            and len(recomputed_missing) == 0
            and len(recomputed_over) == 0
        )
        if self.is_exact_k_cover != expected_exact:
            raise ValueError(
                f"is_exact_k_cover {self.is_exact_k_cover} does not match "
                f"expected {expected_exact} from source"
            )
        return self
