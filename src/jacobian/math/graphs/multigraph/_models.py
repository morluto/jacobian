"""Typed wire contracts for finite multigraph flow and cycle operations.

All operations in this module act on a finite loopless multigraph supplied
with explicit edge IDs so that parallel edges are never conflated.  A finite
Abelian group is represented concretely as a product of cyclic groups with
bounded positive moduli and canonical residue coordinates.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.multigraph._orientation import oriented_endpoints

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
# The shared passive carrier transports the complete simple-graph envelope of
# the declared vertex axis, including SPQR source graphs and skeletons: a
# simple graph on 64 vertices admits up to C(64, 2) = 2016 edges. Individual
# flow and cycle operations retain their own work admission; widening this
# passive value does not widen an exhaustive search claim.
MAX_VERTICES = 64
MAX_EDGES = MAX_VERTICES * (MAX_VERTICES - 1) // 2
MAX_PARALLEL_MULTIPLICITY = 32
MAX_GROUP_RANK = 6
MAX_GROUP_MODULUS = 4096
MAX_GROUP_CARDINALITY = 4096
MAX_FLOW_SEARCH_STATES = 1_048_576
# One Eulerian decomposition splits the admitted edge multiset into cycles of
# at least two edges each, and one simple cycle on the declared vertex axis
# closes with at most MAX_VERTICES + 1 sequence entries.
MAX_CYCLE_COUNT = MAX_EDGES // 2
MAX_CYCLE_LENGTH = MAX_VERTICES + 1
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
            raise PydanticCustomError(
                "graph.multigraph_edges_must_be_loopless_distinct_endpo",
                "multigraph edges must be loopless (distinct endpoints)",
            )
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
                raise PydanticCustomError(
                    "graph.multigraph_edge_ids_must_be_unique",
                    "multigraph edge IDs must be unique",
                )
            seen_ids.add(edge.edge_id)
            if not (
                0 <= edge.left < self.vertex_count
                and 0 <= edge.right < self.vertex_count
            ):
                raise PydanticCustomError(
                    "graph.edge_endpoints_must_be_in_0_vertex_count_1",
                    "edge endpoints must be in 0..vertex_count-1",
                )
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

GroupModulus = Annotated[
    StrictInt,
    Field(ge=2, le=MAX_GROUP_MODULUS),
]


class FiniteAbelianGroup(StrictModel):
    """A finite Abelian group represented as a product of cyclic groups.

    The group is ``(Z/n1Z) x ... x (Z/nrZ)`` with bounded positive moduli.
    Each modulus is an integer in ``2..MAX_GROUP_MODULUS`` and their product
    (the group cardinality) must not exceed ``MAX_GROUP_CARDINALITY``.
    Elements are canonical residue tuples ``(0 <= x_i < n_i)``.  Addition and
    negation are componentwise modulo the respective moduli.  The zero element
    is ``(0, ..., 0)``.
    """

    moduli: tuple[GroupModulus, ...] = Field(
        min_length=1,
        max_length=MAX_GROUP_RANK,
        description=(
            f"Moduli of the cyclic factors, each an integer in "
            f"2..{MAX_GROUP_MODULUS}; their product (the group cardinality) "
            f"must not exceed {MAX_GROUP_CARDINALITY}."
        ),
    )

    @model_validator(mode="after")
    def require_valid_moduli(self) -> Self:
        product = 1
        for modulus in self.moduli:
            if modulus < 2:
                raise PydanticCustomError(
                    "graph.group_moduli_must_be_at_least_2",
                    "group moduli must be at least 2",
                )
            if modulus > MAX_GROUP_MODULUS:
                raise PydanticCustomError(
                    "graph.group_modulus_exceeds_max_group_modulus_product",
                    f"group modulus exceeds {MAX_GROUP_MODULUS}",
                )
            product *= modulus
            if product > MAX_GROUP_CARDINALITY:
                raise PydanticCustomError(
                    "graph.group_cardinality_exceeds_max_group_cardinality_return",
                    f"group cardinality exceeds {MAX_GROUP_CARDINALITY}",
                )
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
            raise PydanticCustomError(
                "graph.element_rank_must_match_group_rank",
                "element rank must match group rank",
            )
        return tuple(
            (a + b) % m for a, b, m in zip(left, right, self.moduli, strict=True)
        )

    def negate(self, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != self.rank:
            raise PydanticCustomError(
                "graph.element_rank_must_match_group_rank",
                "element rank must match group rank",
            )
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
            raise PydanticCustomError(
                "graph.element_rank_must_match_group_rank",
                "element rank must match group rank",
            )
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
                raise PydanticCustomError(
                    "graph.group_element_coordinates_must_be_non_negative",
                    "group element coordinates must be non-negative",
                )
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
    value: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_GROUP_RANK,
        description=(
            "Group element assigned at the tail, one canonical residue per "
            "cyclic factor; the rank must equal the selected group's rank "
            "and each coordinate must satisfy 0 <= coordinate < modulus."
        ),
    )

    @model_validator(mode="after")
    def require_non_negative_value(self) -> Self:
        for coordinate in self.value:
            if coordinate < 0:
                raise PydanticCustomError(
                    "graph.flow_values_must_be_non_negative_residues",
                    "flow values must be non-negative residues",
                )
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
    """Request to check a finite-Abelian flow on a loopless multigraph.

    ``edge_values`` must be a complete assignment: exactly one record per
    graph edge ID, with no repeats and no omissions, and every value must
    be compatible with ``group`` (rank and residue ranges).
    """

    graph: LooplessMultigraph
    group: FiniteAbelianGroup
    edge_values: tuple[FlowEdgeAssignment, ...] = Field(
        max_length=MAX_EDGES,
        description=(
            "Complete oriented flow assignment: exactly one record per "
            "graph edge ID (no repeats, none missing). Each record's value "
            "must have the selected group's rank with coordinates in "
            "0..modulus-1."
        ),
    )

    @model_validator(mode="after")
    def require_complete_assignment(self) -> Self:
        graph_ids = self.graph.edge_id_set
        assigned_ids = {assign.edge_id for assign in self.edge_values}
        if graph_ids != assigned_ids:
            missing = sorted(graph_ids - assigned_ids)
            extra = sorted(assigned_ids - graph_ids)
            raise PydanticCustomError(
                "graph.edge_values_assign_every_edge_exactly_once",
                f"edge_values must assign every edge exactly once; "
                f"missing={missing}, extra={extra}",
            )
        if len(assigned_ids) != len(self.edge_values):
            raise PydanticCustomError(
                "graph.edge_values_must_not_repeat_edge_ids",
                "edge_values must not repeat edge IDs",
            )
        for assign in self.edge_values:
            if len(assign.value) != self.group.rank:
                raise PydanticCustomError(
                    "graph.flow_value_rank_must_match_group_rank",
                    "flow value rank must match group rank",
                )
            if any(
                c >= m for c, m in zip(assign.value, self.group.moduli, strict=True)
            ):
                raise PydanticCustomError(
                    "graph.flow_value_coordinates_must_be_below_their_modul",
                    "flow value coordinates must be below their moduli",
                )
        return self


class MultigraphFlowCheckResult(StrictModel):
    """Exact per-vertex conservation ledger for a submitted flow, bound to its source.

    ``graph`` and ``group`` bind the conclusion to the checked request so the
    ledger, zero-edge set, and booleans remain reconstructible.
    """

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
            raise PydanticCustomError(
                "graph.edge_flow_records_must_cover_exactly_graph_edges",
                "edge_flow_records must cover exactly graph edges",
            )
        if len(record_ids) != len(self.edge_flow_records):
            raise PydanticCustomError(
                "graph.edge_flow_records_must_not_repeat_edge_ids",
                "edge_flow_records must not repeat edge IDs",
            )
        _require_values_within_group(self.edge_flow_records, self.group)
        expected_ledger, expected_conservation = _recompute_divergence_ledger(
            self.graph, self.group, self.edge_flow_records
        )
        if tuple(expected_ledger) != self.divergence_ledger:
            raise PydanticCustomError(
                "graph.divergence_ledger_does_not_match_recomputed_ledg",
                "divergence_ledger does not match recomputed ledger",
            )
        if self.conservation_holds != expected_conservation:
            raise PydanticCustomError(
                "graph.conservation_holds_does_not_match_recomputed_val",
                "conservation_holds does not match recomputed value",
            )
        expected_zero = tuple(
            sorted(
                rec.edge_id
                for rec in self.edge_flow_records
                if self.group.is_zero(rec.value)
            )
        )
        if self.zero_edge_ids != expected_zero:
            raise PydanticCustomError(
                "graph.zero_edge_ids_does_not_match_recomputed_zero_set",
                "zero_edge_ids does not match recomputed zero set",
            )
        if self.nowhere_zero != (len(expected_zero) == 0):
            raise PydanticCustomError(
                "graph.nowhere_zero_does_not_match_zero_edge_set",
                "nowhere_zero does not match zero_edge set",
            )
        return self


def _require_values_within_group(
    records: tuple[FlowEdgeAssignment, ...],
    group: FiniteAbelianGroup,
) -> None:
    for rec in records:
        if len(rec.value) != group.rank:
            raise PydanticCustomError(
                "graph.flow_value_rank_must_match_group_rank",
                "flow value rank must match group rank",
            )
        if any(c >= m for c, m in zip(rec.value, group.moduli, strict=True)):
            raise PydanticCustomError(
                "graph.flow_value_coordinates_must_be_below_moduli",
                "flow value coordinates must be below moduli",
            )
        if any(c < 0 for c in rec.value):
            raise PydanticCustomError(
                "graph.flow_values_must_be_non_negative",
                "flow values must be non-negative",
            )


def _recompute_divergence_ledger(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    records: tuple[FlowEdgeAssignment, ...],
) -> tuple[tuple[VertexDivergence, ...], bool]:
    vertex_out: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    vertex_in: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    vertex_incident: dict[int, set[str]] = {v: set() for v in range(graph.vertex_count)}
    for rec in records:
        edge = graph.edge_by_id(rec.edge_id)
        tail, head = oriented_endpoints(edge, rec.orientation)
        val = group.normalize(rec.value)
        vertex_out[tail].append(val)
        vertex_in[head].append(val)
        vertex_incident[tail].add(rec.edge_id)
        vertex_incident[head].add(rec.edge_id)
    expected_ledger: list[VertexDivergence] = []
    expected_conservation = True
    for v in range(graph.vertex_count):
        out_sum = group.sum(tuple(vertex_out[v]))
        in_sum = group.sum(tuple(vertex_in[v]))
        div = group.add(out_sum, group.negate(in_sum))
        holds = group.is_zero(div)
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
    return tuple(expected_ledger), expected_conservation


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


class _FlowSearchOutcome(StrictModel):
    """Private unbound search outcome used inside the bounded search kernel.

    Carries no retained search domain so the public result type can add the
    required ``graph``/``group``/``resource_budget`` binding without
    recursion.
    """

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


class MultigraphFlowFindResult(StrictModel):
    """Outcome of a bounded finite-Abelian flow search, bound to its source request."""

    graph: LooplessMultigraph
    group: FiniteAbelianGroup
    resource_budget: MultigraphFlowSearchBudget
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
        # The bounded search runs exactly once per request, inside the
        # operation; validation -- including deserialized re-validation --
        # must never repeat it, so the declared budget bounds the total
        # visited states.  Each outcome is therefore checked structurally
        # and arithmetically against the retained source: shape, budget,
        # witness validity, and the exact charged-state totals that only
        # the kernel's termination modes can produce.
        _require_status_shape(self.status, self.flow, self.termination_reason)
        if self.states_explored > self.resource_budget.max_states:
            raise PydanticCustomError(
                "graph.states_explored_exceeds_resource_budget",
                "states_explored exceeds resource budget",
            )
        if self.status == "FOUND":
            if self.termination_reason == "SPECIAL_CASE":
                # SPECIAL_CASE is reserved for the edgeless empty-flow
                # shortcut; the DFS itself never emits it.
                if self.graph.edges or self.flow != ():
                    raise PydanticCustomError(
                        "graph.special_case_termination_requires_edgeless_with_empty",
                        "SPECIAL_CASE termination requires an edgeless "
                        "graph with the empty flow",
                    )
            elif not self.graph.edges:
                raise PydanticCustomError(
                    "graph.edgeless_terminates_as_special_case_termination_reason",
                    "an edgeless graph terminates as SPECIAL_CASE, not "
                    f"{self.termination_reason}",
                )
            assert self.flow is not None  # guaranteed by _require_status_shape
            _verify_found_witness(
                self.graph, self.group, self.resource_budget, self.flow
            )
        elif self.status == "EXHAUSTED":
            if not self.graph.edges:
                raise PydanticCustomError(
                    "graph.exhausted_status_requires_a_nonempty_search_doma",
                    "EXHAUSTED status requires a nonempty search domain",
                )
            expected = _complete_enumeration_state_count(
                self.graph,
                self.group,
                self.resource_budget.require_nowhere_zero,
            )
            if self.states_explored != expected:
                raise PydanticCustomError(
                    "graph.exhausted_outcome_reports_states_explored_explored_states",
                    f"EXHAUSTED outcome reports {self.states_explored} "
                    f"explored states; a completed enumeration of the "
                    f"retained domain charges exactly {expected}",
                )
        else:
            # UNKNOWN is emitted only at the moment the budget is exceeded,
            # reporting exactly max_states charged states.
            if self.states_explored != self.resource_budget.max_states:
                raise PydanticCustomError(
                    "graph.unknown_outcome_reports_states_explored_explored_states",
                    f"UNKNOWN outcome reports {self.states_explored} "
                    "explored states; a budget-exceeded search reports "
                    f"exactly {self.resource_budget.max_states}",
                )
        return self


def _complete_enumeration_state_count(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    require_nowhere_zero: bool,
) -> int:
    """Exact charged-state total of a completed exhaustive flow search.

    The bounded DFS charges one state per pushed partial assignment, so a
    completed enumeration over ``d = len(graph.edges)`` edges with ``b``
    choices per edge visits exactly ``b + b^2 + ... + b^d`` states, where
    ``b`` is twice the number of admissible per-edge values (one term per
    orientation).  Deriving this closed form from the retained source lets
    validators authenticate EXHAUSTED outcomes without repeating the
    bounded search.
    """
    depth = len(graph.edges)
    values = group.cardinality - 1 if require_nowhere_zero else group.cardinality
    branching = 2 * values
    growth: int = branching**depth
    return branching * (growth - 1) // (branching - 1)


def _require_status_shape(
    status: str,
    flow: tuple[FlowEdgeAssignment, ...] | None,
    termination_reason: str,
) -> None:
    """Check the field-shape contract between status, flow, and reason."""

    if status == "FOUND":
        if flow is None:
            raise PydanticCustomError(
                "graph.found_status_requires_a_flow_witness",
                "FOUND status requires a flow witness",
            )
        if termination_reason not in ("WITNESS_FOUND", "SPECIAL_CASE"):
            raise PydanticCustomError(
                "graph.found_status_requires_witness_found_special_case",
                "FOUND status requires WITNESS_FOUND or SPECIAL_CASE reason",
            )
    elif status == "EXHAUSTED":
        if flow is not None:
            raise PydanticCustomError(
                "graph.exhausted_status_must_not_include_a_flow",
                "EXHAUSTED status must not include a flow",
            )
        if termination_reason != "SEARCH_EXHAUSTED":
            raise PydanticCustomError(
                "graph.exhausted_status_requires_search_exhausted_reaso",
                "EXHAUSTED status requires SEARCH_EXHAUSTED reason",
            )
    elif status == "UNKNOWN":
        if flow is not None:
            raise PydanticCustomError(
                "graph.unknown_status_must_not_include_a_flow",
                "UNKNOWN status must not include a flow",
            )
        if termination_reason != "STATE_BUDGET_EXCEEDED":
            raise PydanticCustomError(
                "graph.unknown_status_requires_state_budget_exceeded",
                "UNKNOWN status requires STATE_BUDGET_EXCEEDED",
            )


def _require_single_assignment_per_edge(
    flow: tuple[FlowEdgeAssignment, ...],
    graph_ids: frozenset[str],
) -> None:
    """Require exactly one assignment record per graph edge."""

    if len(flow) != len(graph_ids):
        raise PydanticCustomError(
            "graph.found_flow_must_assign_exactly_one_value_to_ever",
            "FOUND flow must assign exactly one value to every graph edge",
        )
    assigned: set[str] = set()
    for a in flow:
        if a.edge_id in assigned:
            raise PydanticCustomError(
                "graph.found_flow_assigns_edge_edge_id_more",
                f"FOUND flow assigns edge {a.edge_id} more than once",
            )
        assigned.add(a.edge_id)
    if graph_ids != assigned:
        raise PydanticCustomError(
            "graph.found_flow_must_assign_every_graph_edge",
            "FOUND flow must assign every graph edge",
        )


def _verify_found_witness(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    resource_budget: MultigraphFlowSearchBudget,
    flow: tuple[FlowEdgeAssignment, ...],
) -> None:
    """Recompute coverage, nowhere-zero, and conservation for a FOUND witness."""

    graph_ids = graph.edge_id_set
    _require_single_assignment_per_edge(flow, graph_ids)
    if resource_budget.require_nowhere_zero:
        for a in flow:
            if group.is_zero(a.value):
                raise PydanticCustomError(
                    "graph.found_flow_violates_nowhere_zero_requirement",
                    "FOUND flow violates nowhere_zero requirement",
                )
    vertex_out: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    vertex_in: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    for a in flow:
        edge = graph.edge_by_id(a.edge_id)
        tail, head = oriented_endpoints(edge, a.orientation)
        val = group.normalize(a.value)
        vertex_out[tail].append(val)
        vertex_in[head].append(val)
    for v in range(graph.vertex_count):
        out_sum = group.sum(tuple(vertex_out[v]))
        in_sum = group.sum(tuple(vertex_in[v]))
        if not group.is_zero(group.add(out_sum, group.negate(in_sum))):
            raise PydanticCustomError(
                "graph.found_flow_does_not_satisfy_conservation",
                "FOUND flow does not satisfy conservation",
            )


# ---------------------------------------------------------------------------
# Eulerian cycle decomposition
# ---------------------------------------------------------------------------


class EulerianCyclesRequest(StrictModel):
    """Request to decompose an edge multiset into edge-disjoint cycles.

    When ``edge_subset`` is omitted, the full edge set of ``graph`` is used.
    When provided, it must reference existing edge IDs without repeats; any
    induced-degree parity is accepted.  Parity determines the result: an
    Eulerian (all even induced degrees) multiset returns one deterministic
    full decomposition with ``covers_all=True``, while a non-Eulerian
    multiset returns the empty decomposition with ``covers_all=False``.
    """

    graph: LooplessMultigraph
    edge_subset: tuple[StrictStr, ...] | None = Field(
        default=None,
        max_length=MAX_EDGES,
        description=(
            "Optional edge IDs to decompose; must reference existing edge IDs "
            "without repeats. Any induced-degree parity is accepted. Even "
            "(Eulerian) parity yields a full decomposition with covers_all=True; "
            "odd parity yields the empty decomposition with covers_all=False."
        ),
    )

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        if self.edge_subset is not None:
            graph_ids = self.graph.edge_id_set
            subset_set = set(self.edge_subset)
            if not subset_set.issubset(graph_ids):
                missing = sorted(subset_set - graph_ids)
                raise PydanticCustomError(
                    "graph.edge_subset_references_unknown_edge_ids_missing",
                    f"edge_subset references unknown edge IDs: {missing}",
                )
            if len(subset_set) != len(self.edge_subset):
                raise PydanticCustomError(
                    "graph.edge_subset_must_not_repeat_edge_ids",
                    "edge_subset must not repeat edge IDs",
                )
        return self


class CycleRecord(StrictModel):
    """One explicit cycle as a closed alternating vertex/edge-ID sequence.

    ``vertices`` and ``edge_ids`` alternate so that ``edge_ids[i]`` connects
    ``vertices[i]`` and ``vertices[(i+1) % len]``.  The sequence is closed:
    the first vertex equals the last, and ``len(vertices) == len(edge_ids) + 1``.
    It is a simple cycle: every vertex except the repeated closing endpoint
    must be distinct.  A closed walk that revisits an interior vertex (for
    example a figure-eight made of two cycles sharing one vertex) is not a
    cycle and is rejected.
    """

    vertices: tuple[StrictInt, ...] = Field(
        min_length=3,
        max_length=MAX_CYCLE_LENGTH,
        description=(
            "Closed vertex sequence: first equals last, len(vertices) == "
            "len(edge_ids) + 1, and every interior vertex (all entries except "
            "the closing duplicate) must be distinct."
        ),
    )
    edge_ids: tuple[StrictStr, ...] = Field(
        min_length=2,
        max_length=MAX_VERTICES,
        description=(
            "Edge IDs alternating with vertices: edge_ids[i] connects "
            "vertices[i] and vertices[i+1]; no edge ID may repeat."
        ),
    )

    @model_validator(mode="after")
    def require_closed_cycle(self) -> Self:
        if len(self.vertices) != len(self.edge_ids) + 1:
            raise PydanticCustomError(
                "graph.a_cycle_must_have_len_vertices_len_edge_ids_1",
                "a cycle must have len(vertices) == len(edge_ids) + 1",
            )
        if self.vertices[0] != self.vertices[-1]:
            raise PydanticCustomError(
                "graph.a_cycle_must_be_closed_first_vertex_last",
                "a cycle must be closed (first vertex == last)",
            )
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise PydanticCustomError(
                "graph.a_cycle_must_not_repeat_edge_ids",
                "a cycle must not repeat edge IDs",
            )
        if any(v < 0 or v >= MAX_VERTICES for v in self.vertices):
            raise PydanticCustomError(
                "graph.cycle_vertices_max_vertices_if_len_set",
                f"cycle vertices must be in 0..{MAX_VERTICES - 1}",
            )
        if len(set(self.vertices[:-1])) != len(self.vertices) - 1:
            raise PydanticCustomError(
                "graph.cycle_repeat_interior_vertices_all_vertices_except",
                "a cycle must not repeat interior vertices "
                "(all vertices except the closing duplicate must be distinct)",
            )
        return self


class EulerianCyclesResult(StrictModel):
    """Deterministic edge-disjoint cycle decomposition, bound to its source edge set.

    Validation replays the defining relation against the retained source:
    incidence, disjointness, usage, coverage, and the Eulerian dichotomy —
    an Eulerian requested multiset must be fully decomposed with
    ``covers_all=True``, while the empty ``covers_all=False`` outcome is
    reserved for a non-Eulerian requested multiset.
    """

    graph: LooplessMultigraph
    edge_subset: tuple[StrictStr, ...] | None = Field(
        default=None, max_length=MAX_EDGES
    )
    cycles: tuple[CycleRecord, ...] = Field(max_length=MAX_CYCLE_COUNT)
    edge_usage: tuple[tuple[StrictStr, StrictInt], ...] = Field(max_length=MAX_EDGES)
    covers_all: bool

    @model_validator(mode="after")
    def require_bound_eulerian(self) -> Self:
        # Determine requested edge IDs; apply the request's uniqueness check
        # before any set conversion so duplicates cannot silently change the
        # purported multiset's multiplicity.
        if self.edge_subset is not None:
            if len(set(self.edge_subset)) != len(self.edge_subset):
                raise PydanticCustomError(
                    "graph.edge_subset_must_not_repeat_edge_ids",
                    "edge_subset must not repeat edge IDs",
                )
            requested = set(self.edge_subset)
            if not requested.issubset(self.graph.edge_id_set):
                raise PydanticCustomError(
                    "graph.edge_subset_contains_unknown_edge_ids",
                    "edge_subset contains unknown edge IDs",
                )
        else:
            requested = set(self.graph.edge_id_set)
        used = _verify_cycle_incidence(self.graph, self.cycles, requested)
        # Validate edge_usage matches
        expected_usage = tuple(sorted((eid, used[eid]) for eid in sorted(requested)))
        if self.edge_usage != expected_usage:
            raise PydanticCustomError(
                "graph.edge_usage_edge_usage_does_match_recomputed",
                f"edge_usage {self.edge_usage} does not match recomputed {expected_usage}",
            )
        expected_covers = _expected_covers_all(used, requested, len(self.cycles))
        if self.covers_all != expected_covers:
            raise PydanticCustomError(
                "graph.covers_all_covers_all_does_match_expected",
                f"covers_all {self.covers_all} does not match expected {expected_covers}",
            )
        if requested and _subset_is_eulerian(self.graph, requested):
            # An Eulerian source must be fully decomposed; the empty
            # covers_all=False outcome is reserved for non-Eulerian sources.
            if not self.covers_all:
                raise PydanticCustomError(
                    "graph.eulerian_source_fully_decomposed_covers_all_empty",
                    "an Eulerian source must be fully decomposed "
                    "(covers_all=True); an empty covers_all=False outcome "
                    "is only valid for a non-Eulerian source",
                )
        elif requested and self.cycles != ():
            raise PydanticCustomError(
                "graph.non_eulerian_source_yield_empty_decomposition_cycles",
                "a non-Eulerian source must yield an empty decomposition "
                "(cycles=()) with covers_all=False",
            )
        return self


def _verify_cycle_incidence(
    graph: LooplessMultigraph,
    cycles: tuple[CycleRecord, ...],
    requested: set[str],
) -> dict[str, int]:
    """Check edge-disjointness and endpoint incidence; return per-edge usage."""

    used: dict[str, int] = dict.fromkeys(requested, 0)
    seen_cycle_ids: set[str] = set()
    for cycle in cycles:
        for eid in cycle.edge_ids:
            if eid not in requested:
                raise PydanticCustomError(
                    "graph.cycle_uses_edge_eid_requested_set_if",
                    f"cycle uses edge {eid} not in requested set",
                )
            if eid in seen_cycle_ids:
                raise PydanticCustomError(
                    "graph.edge_eid_appears_multiple_cycles_disjoint",
                    f"edge {eid} appears in multiple cycles (must be disjoint)",
                )
            seen_cycle_ids.add(eid)
            pos = list(cycle.edge_ids).index(eid)
            v_from = cycle.vertices[pos]
            v_to = cycle.vertices[pos + 1]
            edge = graph.edge_by_id(eid)
            if not (
                (edge.left == v_from and edge.right == v_to)
                or (edge.right == v_from and edge.left == v_to)
            ):
                raise PydanticCustomError(
                    "graph.cycle_edge_eid_incidence_does_match_used",
                    f"cycle edge {eid} incidence does not match graph",
                )
            used[eid] += 1
    return used


def _subset_is_eulerian(
    graph: LooplessMultigraph,
    requested: set[str],
) -> bool:
    """Return whether every vertex has even degree in the requested multiset."""

    degree: dict[int, int] = dict.fromkeys(range(graph.vertex_count), 0)
    for eid in requested:
        edge = graph.edge_by_id(eid)
        degree[edge.left] += 1
        degree[edge.right] += 1
    return all(d % 2 == 0 for d in degree.values())


def _expected_covers_all(
    used: dict[str, int],
    requested: set[str],
    cycle_count: int,
) -> bool:
    if len(requested) == 0:
        return True
    return all(used[eid] == 1 for eid in requested) and (
        len(requested) > 0 or cycle_count == 0
    )


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
            raise PydanticCustomError(
                "graph.total_cycle_edge_incidences_exceed_max_cycle",
                f"total cycle-edge incidences exceed {MAX_CYCLE_EDGE_INCIDENCES}",
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
    def require_consistent_cover(self) -> Self:
        if len(self.cycle_validity) != len(self.cycles):
            raise PydanticCustomError(
                "graph.cycle_validity_length_must_match_cycles_length",
                "cycle_validity length must match cycles length",
            )
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
            raise PydanticCustomError(
                "graph.cycle_validity_cycle_validity_does_match_recomputed",
                f"cycle_validity {self.cycle_validity} does not match "
                f"recomputed {tuple(recomputed_validity)} from source",
            )
        recomputed_tuple = tuple(
            (eid, recomputed_multiplicity[eid])
            for eid in sorted(recomputed_multiplicity)
        )
        if recomputed_tuple != self.edge_multiplicity:
            raise PydanticCustomError(
                "graph.edge_multiplicity_edge_multiplicity_does_match_recomputed",
                f"edge_multiplicity {self.edge_multiplicity} does not match "
                f"recomputed {recomputed_tuple} from source",
            )
        k = self.target_multiplicity
        recomputed_missing = tuple(
            sorted(eid for eid, cnt in recomputed_multiplicity.items() if cnt < k)
        )
        recomputed_over = tuple(
            sorted(eid for eid, cnt in recomputed_multiplicity.items() if cnt > k)
        )
        if recomputed_missing != self.missing_edge_ids:
            raise PydanticCustomError(
                "graph.missing_edge_ids_missing_edge_ids_does",
                f"missing_edge_ids {self.missing_edge_ids} does not match "
                f"recomputed {recomputed_missing} for k={k}",
            )
        if recomputed_over != self.overcovered_edge_ids:
            raise PydanticCustomError(
                "graph.overcovered_edge_ids_overcovered_edge_ids_does",
                f"overcovered_edge_ids {self.overcovered_edge_ids} does not match "
                f"recomputed {recomputed_over} for k={k}",
            )
        expected_exact = (
            all(recomputed_validity)
            and len(recomputed_missing) == 0
            and len(recomputed_over) == 0
        )
        if self.is_exact_k_cover != expected_exact:
            raise PydanticCustomError(
                "graph.exact_cover_exact_cover_does_match_expected",
                f"is_exact_k_cover {self.is_exact_k_cover} does not match "
                f"expected {expected_exact} from source",
            )
        return self
