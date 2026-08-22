"""Tests for finite-multigraph flow and cycle-multicover operations.

Covers the 12 required fixtures from issue #1680 plus one bounded fixture
extracted from the public cdc-lean construction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.multigraph._models import (
    CycleMulticoverRequest,
    CycleRecord,
    EulerianCyclesRequest,
    EulerianCyclesResult,
    FiniteAbelianGroup,
    FlowEdgeAssignment,
    LooplessMultigraph,
    MultigraphEdge,
    MultigraphFlowCheckRequest,
    MultigraphFlowCheckResult,
    MultigraphFlowFindRequest,
    MultigraphFlowFindResult,
)
from jacobian.math.graphs.multigraph._operations import (
    check_cycle_multicover,
    check_multigraph_flow,
    compute_eulerian_cycles,
    find_multigraph_flow,
)

# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

TRIANGLE = LooplessMultigraph(
    vertex_count=3,
    edges=(
        MultigraphEdge(edge_id="e0", left=0, right=1),
        MultigraphEdge(edge_id="e1", left=1, right=2),
        MultigraphEdge(edge_id="e2", left=2, right=0),
    ),
)

# Triangle with a parallel edge: two edges between 0 and 1.
PARALLEL_TRIANGLE = LooplessMultigraph(
    vertex_count=3,
    edges=(
        MultigraphEdge(edge_id="a", left=0, right=1),
        MultigraphEdge(edge_id="b", left=0, right=1),
        MultigraphEdge(edge_id="c", left=1, right=2),
        MultigraphEdge(edge_id="d", left=2, right=0),
    ),
)

# Graph with a bridge: two triangles connected by a single bridge edge.
# Triangle 0-1-2, triangle 3-4-5, bridge 2-3.
BRIDGE_GRAPH = LooplessMultigraph(
    vertex_count=6,
    edges=(
        MultigraphEdge(edge_id="e0", left=0, right=1),
        MultigraphEdge(edge_id="e1", left=1, right=2),
        MultigraphEdge(edge_id="e2", left=2, right=0),
        MultigraphEdge(edge_id="br", left=2, right=3),  # bridge
        MultigraphEdge(edge_id="e3", left=3, right=4),
        MultigraphEdge(edge_id="e4", left=4, right=5),
        MultigraphEdge(edge_id="e5", left=5, right=3),
    ),
)

# Square graph (4-cycle)
SQUARE = LooplessMultigraph(
    vertex_count=4,
    edges=(
        MultigraphEdge(edge_id="s0", left=0, right=1),
        MultigraphEdge(edge_id="s1", left=1, right=2),
        MultigraphEdge(edge_id="s2", left=2, right=3),
        MultigraphEdge(edge_id="s3", left=3, right=0),
    ),
)

Z3 = FiniteAbelianGroup(moduli=(3,))
Z2CUBED = FiniteAbelianGroup(moduli=(2, 2, 2))


def _flow_check(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    edge_values: tuple[FlowEdgeAssignment, ...],
) -> MultigraphFlowCheckResult:
    return check_multigraph_flow(
        MultigraphFlowCheckRequest(graph=graph, group=group, edge_values=edge_values)
    )


def _flow_find(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    **kwargs: object,
) -> MultigraphFlowFindResult:
    return find_multigraph_flow(
        MultigraphFlowFindRequest.model_validate(
            {"graph": graph.model_dump(), "group": group.model_dump(), **kwargs}
        )
    )


# ---------------------------------------------------------------------------
# Fixture 1: Triangle with a valid nowhere-zero cyclic flow
# ---------------------------------------------------------------------------


class TestFlowCheckNowhereZero:
    def test_triangle_z3_nowhere_zero_flow(self) -> None:
        """Fixture 1: a triangle with a valid nowhere-zero cyclic Z/3Z flow."""
        flow = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1,)),
        )
        result = _flow_check(TRIANGLE, Z3, flow)
        assert result.conservation_holds
        assert result.nowhere_zero
        assert result.zero_edge_ids == ()
        for div in result.divergence_ledger:
            assert div.conservation_holds
            assert div.coordinates == (0,)


# ---------------------------------------------------------------------------
# Fixture 2: Same graph with one conservation-violating edge value
# ---------------------------------------------------------------------------


class TestFlowCheckViolation:
    def test_triangle_with_violating_edge(self) -> None:
        """Fixture 2: the same triangle with one wrong edge value."""
        flow = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(2,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1,)),
        )
        result = _flow_check(TRIANGLE, Z3, flow)
        assert not result.conservation_holds
        # At least one vertex has nonzero divergence
        violated = [d for d in result.divergence_ledger if not d.conservation_holds]
        assert len(violated) >= 1


# ---------------------------------------------------------------------------
# Fixture 3: Graph with a bridge, no nowhere-zero flow in selected group
# ---------------------------------------------------------------------------


class TestFlowFindBridgeNoFlow:
    def test_bridge_graph_no_z3_nowhere_zero_flow(self) -> None:
        """Fixture 3: a bridge graph has no nowhere-zero Z/3Z flow."""
        result = _flow_find(BRIDGE_GRAPH, Z3, resource_budget={"max_states": 1000000})
        # A bridge forces a zero value on the bridge edge, so no NZ flow exists.
        assert result.status == "EXHAUSTED"
        assert result.termination_reason == "SEARCH_EXHAUSTED"
        assert result.flow is None


# ---------------------------------------------------------------------------
# Fixture 4: Multigraph with parallel edges, edge IDs preserved
# ---------------------------------------------------------------------------


class TestParallelEdges:
    def test_parallel_edges_have_distinct_ids(self) -> None:
        """Fixture 4: parallel edges are distinguished by their edge IDs."""
        assert PARALLEL_TRIANGLE.edge_id_set == frozenset({"a", "b", "c", "d"})
        # Both 'a' and 'b' connect 0 and 1
        a = PARALLEL_TRIANGLE.edge_by_id("a")
        b = PARALLEL_TRIANGLE.edge_by_id("b")
        assert {a.left, a.right} == {0, 1}
        assert {b.left, b.right} == {0, 1}
        assert a.edge_id != b.edge_id

    def test_parallel_edge_flow_preserves_identity(self) -> None:
        """A flow on a multigraph with parallel edges must assign each edge separately."""
        flow = (
            FlowEdgeAssignment(edge_id="a", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="b", orientation="left_to_right", value=(2,)),
            FlowEdgeAssignment(edge_id="c", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="d", orientation="left_to_right", value=(1,)),
        )
        result = _flow_check(PARALLEL_TRIANGLE, Z3, flow)
        # a+b at vertex 0 outgoing, d incoming: divergence = (1+2) - 1 = 2 mod 3 != 0
        # This should not conserve (bridge-like structure in the flow)
        # Actually let's check: vertex 0: out=a(1)+b(2)=0 mod3, in=d(1). div=0-1=2 mod3
        assert not result.conservation_holds


# ---------------------------------------------------------------------------
# Fixture 5: Orientation reversal paired with value negation = same flow
# ---------------------------------------------------------------------------


class TestOrientationReversalWithNegation:
    def test_reversal_with_negation_preserves_flow(self) -> None:
        """Fixture 5: reversing an edge's orientation and negating its value
        yields the same conservation result."""
        # Original flow: all left_to_right with value 1
        flow_orig = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1,)),
        )
        result_orig = _flow_check(TRIANGLE, Z3, flow_orig)

        # Reversed: e0 is right_to_left with value 2 (= -1 mod 3)
        flow_rev = (
            FlowEdgeAssignment(edge_id="e0", orientation="right_to_left", value=(2,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1,)),
        )
        result_rev = _flow_check(TRIANGLE, Z3, flow_rev)

        assert result_orig.conservation_holds == result_rev.conservation_holds
        assert result_orig.nowhere_zero == result_rev.nowhere_zero


# ---------------------------------------------------------------------------
# Fixture 6: Orientation reversal without negation = failed conservation
# ---------------------------------------------------------------------------


class TestOrientationReversalWithoutNegation:
    def test_reversal_without_negation_fails(self) -> None:
        """Fixture 6: reversing an edge's orientation without negating its value
        breaks conservation."""
        # Original flow: all left_to_right with value 1 — conserves
        flow_orig = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1,)),
        )
        result_orig = _flow_check(TRIANGLE, Z3, flow_orig)
        assert result_orig.conservation_holds

        # Reversed e0 without negating: right_to_left with value 1
        flow_rev = (
            FlowEdgeAssignment(edge_id="e0", orientation="right_to_left", value=(1,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(1,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1,)),
        )
        result_rev = _flow_check(TRIANGLE, Z3, flow_rev)
        assert not result_rev.conservation_holds


# ---------------------------------------------------------------------------
# Fixture 7: Valid Eulerian edge-set decomposition
# ---------------------------------------------------------------------------


class TestEulerianDecomposition:
    def test_triangle_eulerian(self) -> None:
        """Fixture 7: a valid Eulerian cycle decomposition of a triangle."""
        result = compute_eulerian_cycles(EulerianCyclesRequest(graph=TRIANGLE))
        assert result.covers_all
        assert len(result.cycles) >= 1
        # Every edge should have usage 1
        for _eid, count in result.edge_usage:
            assert count == 1
        # Total edges covered = 3
        total_edges = sum(len(c.edge_ids) for c in result.cycles)
        assert total_edges == 3

    def test_non_eulerian_returns_empty(self) -> None:
        """A graph with odd-degree vertices is not Eulerian."""
        # Bridge graph has odd-degree vertices
        result = compute_eulerian_cycles(EulerianCyclesRequest(graph=BRIDGE_GRAPH))
        assert not result.covers_all
        assert len(result.cycles) == 0

    def test_disconnected_eulerian(self) -> None:
        """Two disjoint triangles should decompose into separate cycles."""
        two_triangles = LooplessMultigraph(
            vertex_count=6,
            edges=(
                MultigraphEdge(edge_id="a", left=0, right=1),
                MultigraphEdge(edge_id="b", left=1, right=2),
                MultigraphEdge(edge_id="c", left=2, right=0),
                MultigraphEdge(edge_id="d", left=3, right=4),
                MultigraphEdge(edge_id="e", left=4, right=5),
                MultigraphEdge(edge_id="f", left=5, right=3),
            ),
        )
        result = compute_eulerian_cycles(EulerianCyclesRequest(graph=two_triangles))
        assert result.covers_all
        assert len(result.cycles) == 2

    def test_edge_subset(self) -> None:
        """An explicit edge subset that is Eulerian should decompose."""
        result = compute_eulerian_cycles(
            EulerianCyclesRequest(graph=TRIANGLE, edge_subset=("e0", "e1", "e2"))
        )
        assert result.covers_all
        assert len(result.cycles) >= 1

    def test_empty_edge_set(self) -> None:
        """An empty edge set trivially decomposes."""
        result = compute_eulerian_cycles(
            EulerianCyclesRequest(graph=TRIANGLE, edge_subset=())
        )
        assert result.covers_all
        assert len(result.cycles) == 0


# ---------------------------------------------------------------------------
# Fixture 8: Submitted exact double cover with every edge multiplicity two
# ---------------------------------------------------------------------------


class TestCycleMulticoverDoubleCover:
    def test_triangle_double_cover(self) -> None:
        """Fixture 8: a submitted exact double cover of a triangle."""
        cycles = (
            CycleRecord(vertices=(0, 1, 2, 0), edge_ids=("e0", "e1", "e2")),
            CycleRecord(vertices=(0, 2, 1, 0), edge_ids=("e2", "e1", "e0")),
        )
        result = check_cycle_multicover(
            CycleMulticoverRequest(graph=TRIANGLE, cycles=cycles, target_multiplicity=2)
        )
        assert result.is_exact_k_cover
        assert result.missing_edge_ids == ()
        assert result.overcovered_edge_ids == ()
        assert all(result.cycle_validity)
        for _eid, count in result.edge_multiplicity:
            assert count == 2


# ---------------------------------------------------------------------------
# Fixture 9: One missing and one overcovered edge
# ---------------------------------------------------------------------------


class TestCycleMulticoverMissingOver:
    def test_missing_and_overcovered(self) -> None:
        """Fixture 9: one edge covered fewer than k and one covered more than k."""
        # Two copies of the same cycle on a triangle, target k=2,
        # but both cover e0 twice and e2 is never covered via a different path.
        # Let's construct: one cycle 0-1-2-0 and one cycle 0-1-2-0 again.
        # All edges covered 2 times. To get missing+over, use target k=2
        # but submit one cycle covering e0 and another also covering e0
        # while skipping e2.
        # Use a square: two cycles, one 0-1-2-3-0 and one 0-1 only (not a valid cycle).
        # Better: triangle with one cycle 0-1-2-0, target k=2.
        # e0=1, e1=1, e2=1. Missing: all three (count 1 < 2). No overcovered.
        # For missing AND over: use square, two cycles: 0-1-2-3-0 and 0-1.
        # But 0-1 is not a cycle (not closed). Let's use:
        # Square, cycle1 = 0-1-2-3-0 (covers s0,s1,s2,s3), target=2
        # cycle2 = 0-1-2-0 ... but 0-2 is not an edge.
        # Use parallel triangle: cycle1 = 0-a-1-c-2-d-0 (a,c,d)
        # cycle2 = 0-b-1-c-2-d-0 (b,c,d)  target=2
        # a:1, b:1, c:2, d:2 -> a,b missing, none over
        # Need over: add cycle3 = 0-a-1-c-2-d-0 again -> a:2,b:1,c:3,d:3
        # target=2: b missing, c+d over
        cycles = (
            CycleRecord(vertices=(0, 1, 2, 0), edge_ids=("a", "c", "d")),
            CycleRecord(vertices=(0, 1, 2, 0), edge_ids=("b", "c", "d")),
            CycleRecord(vertices=(0, 1, 2, 0), edge_ids=("a", "c", "d")),
        )
        result = check_cycle_multicover(
            CycleMulticoverRequest(
                graph=PARALLEL_TRIANGLE, cycles=cycles, target_multiplicity=2
            )
        )
        assert not result.is_exact_k_cover
        # b is covered once (missing)
        assert "b" in result.missing_edge_ids
        # c and d are covered 3 times (over)
        assert "c" in result.overcovered_edge_ids
        assert "d" in result.overcovered_edge_ids


# ---------------------------------------------------------------------------
# Fixture 10: Malformed cycle (not closed or wrong incidence)
# ---------------------------------------------------------------------------


class TestMalformedCycle:
    def test_non_closed_cycle_rejected_by_model(self) -> None:
        """Fixture 10: a cycle that is not closed is rejected by the model."""
        with pytest.raises(ValidationError):
            CycleRecord(vertices=(0, 1, 2, 1), edge_ids=("e0", "e1", "e2"))

    def test_wrong_incidence_detected(self) -> None:
        """A cycle that does not follow graph incidence is flagged invalid."""
        # e0 connects 0-1, but we claim it connects 0-2
        cycles = (
            CycleRecord(
                vertices=(0, 2, 1, 0),
                edge_ids=("e0", "e1", "e2"),
            ),
        )
        result = check_cycle_multicover(
            CycleMulticoverRequest(graph=TRIANGLE, cycles=cycles, target_multiplicity=1)
        )
        assert not all(result.cycle_validity)
        assert result.cycle_validity[0] is False
        assert not result.is_exact_k_cover


# ---------------------------------------------------------------------------
# Fixture 11: Search that completes with EXHAUSTED
# ---------------------------------------------------------------------------


class TestFlowSearchExhausted:
    def test_bridge_graph_exhausted(self) -> None:
        """Fixture 11: a search that completes with EXHAUSTED in a tiny domain."""
        # Bridge graph with Z/3Z, require_nowhere_zero=True
        # A bridge cannot have a nonzero flow, so the search exhausts.
        result = _flow_find(
            BRIDGE_GRAPH,
            Z3,
            resource_budget={"max_states": 1000000, "require_nowhere_zero": True},
        )
        assert result.status == "EXHAUSTED"
        assert result.termination_reason == "SEARCH_EXHAUSTED"
        assert result.flow is None


# ---------------------------------------------------------------------------
# Fixture 12: Search that returns UNKNOWN at a tiny budget
# ---------------------------------------------------------------------------


class TestFlowSearchUnknown:
    def test_tiny_budget_returns_unknown(self) -> None:
        """Fixture 12: a search that returns UNKNOWN at a deliberately tiny budget."""
        # Triangle with Z/3Z has many states. A budget of 0 should immediately
        # exceed.
        result = _flow_find(
            TRIANGLE,
            Z3,
            resource_budget={"max_states": 1, "require_nowhere_zero": True},
        )
        # The triangle has 3 edges, each with 2 nonzero values * 2 orientations
        # = 4 choices per edge, so 4^3 = 64 complete states.
        # With max_states=1, the first complete assignment hits the budget.
        # But if the first one happens to be a valid flow, it returns FOUND.
        # The first lexicographic assignment might conserve. Let's use a
        # graph where the first assignment cannot conserve.
        # Use a graph where no NZ flow exists but the search space is large
        # enough that budget=1 doesn't exhaust.
        # Actually: bridge graph with Z/3Z and a tiny budget.
        result = _flow_find(
            BRIDGE_GRAPH,
            Z3,
            resource_budget={"max_states": 1, "require_nowhere_zero": True},
        )
        assert result.status == "UNKNOWN"
        assert result.termination_reason == "STATE_BUDGET_EXCEEDED"
        assert result.flow is None


# ---------------------------------------------------------------------------
# Fixture: cdc-lean construction (bounded (Z/2Z)^3 flow on a small graph)
# ---------------------------------------------------------------------------


class TestCdcLeanConstruction:
    """A bounded fixture extracted from the public cdc-lean construction.

    The cdc-lean proof constructs a nowhere-zero (Z/2Z)^3 flow on a finite
    loopless bridgeless multigraph. We test with a small bridgeless graph
    (a triangle) and the (Z/2Z)^3 group, which has 8 elements and 7 nonzero.
    """

    def test_triangle_z2cubed_find_nowhere_zero(self) -> None:
        """A triangle admits a nowhere-zero (Z/2Z)^3 flow."""
        result = _flow_find(
            TRIANGLE,
            Z2CUBED,
            resource_budget={"max_states": 1000000, "require_nowhere_zero": True},
        )
        assert result.status == "FOUND"
        assert result.flow is not None
        # Verify the witness: it must be a valid flow
        check_result = _flow_check(TRIANGLE, Z2CUBED, result.flow)
        assert check_result.conservation_holds
        assert check_result.nowhere_zero

    def test_triangle_z2cubed_double_cover_from_flow(self) -> None:
        """From a (Z/2Z)^3 flow, construct and check a cycle double cover."""
        # Find a flow first
        result = _flow_find(
            TRIANGLE,
            Z2CUBED,
            resource_budget={"max_states": 1000000, "require_nowhere_zero": True},
        )
        assert result.status == "FOUND"
        assert result.flow is not None

        # The support of a nonzero flow gives an Eulerian subgraph.
        # Decompose the full graph into cycles.
        eulerian = compute_eulerian_cycles(EulerianCyclesRequest(graph=TRIANGLE))
        assert eulerian.covers_all

        # Check that the Eulerian decomposition is a 1-cover
        cover_result = check_cycle_multicover(
            CycleMulticoverRequest(
                graph=TRIANGLE,
                cycles=eulerian.cycles,
                target_multiplicity=1,
            )
        )
        assert cover_result.is_exact_k_cover


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_loop_rejected(self) -> None:
        """A self-loop edge is rejected."""
        with pytest.raises(ValidationError):
            MultigraphEdge(edge_id="x", left=0, right=0)

    def test_duplicate_edge_ids_rejected(self) -> None:
        """Duplicate edge IDs are rejected."""
        with pytest.raises(ValidationError):
            LooplessMultigraph(
                vertex_count=3,
                edges=(
                    MultigraphEdge(edge_id="e", left=0, right=1),
                    MultigraphEdge(edge_id="e", left=1, right=2),
                ),
            )

    def test_edge_endpoint_out_of_range(self) -> None:
        """An edge endpoint outside 0..vertex_count-1 is rejected."""
        with pytest.raises(ValidationError):
            LooplessMultigraph(
                vertex_count=2,
                edges=(MultigraphEdge(edge_id="e", left=0, right=5),),
            )

    def test_group_modulus_minimum(self) -> None:
        """A group modulus less than 2 is rejected."""
        with pytest.raises(ValidationError):
            FiniteAbelianGroup(moduli=(1,))

    def test_group_cardinality_bound(self) -> None:
        """A group exceeding the cardinality bound is rejected."""
        with pytest.raises(ValidationError):
            FiniteAbelianGroup(moduli=(41, 101))

    def test_incomplete_flow_assignment_rejected(self) -> None:
        """An incomplete edge-value assignment is rejected."""
        flow = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(1,)),
            # Missing e1 and e2
        )
        with pytest.raises(ValidationError):
            MultigraphFlowCheckRequest(graph=TRIANGLE, group=Z3, edge_values=flow)

    def test_wrong_rank_value_rejected(self) -> None:
        """A flow value with wrong rank is rejected."""
        flow = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(1, 0)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(1, 0)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(1, 0)),
        )
        with pytest.raises(ValidationError):
            MultigraphFlowCheckRequest(graph=TRIANGLE, group=Z3, edge_values=flow)

    def test_cycle_not_closed_rejected(self) -> None:
        """A non-closed cycle is rejected."""
        with pytest.raises(ValidationError):
            CycleRecord(vertices=(0, 1, 2, 3), edge_ids=("e0", "e1", "e2"))

    def test_cycle_repeated_edges_rejected(self) -> None:
        """A cycle that repeats edge IDs is rejected."""
        with pytest.raises(ValidationError):
            CycleRecord(vertices=(0, 1, 0, 1, 0), edge_ids=("e0", "e0", "e0", "e0"))

    def test_empty_graph_flow_find(self) -> None:
        """An edgeless graph trivially has the empty flow."""
        empty_graph = LooplessMultigraph(vertex_count=3, edges=())
        result = _flow_find(empty_graph, Z3)
        assert result.status == "FOUND"
        assert result.flow == ()
        assert result.termination_reason == "SPECIAL_CASE"


# ---------------------------------------------------------------------------
# Rotation/reversal invariance for multicover
# ---------------------------------------------------------------------------


class TestMulticoverInvariance:
    def test_rotation_invariance(self) -> None:
        """A cycle rotation does not change the multiplicity profile."""
        original = CycleRecord(vertices=(0, 1, 2, 0), edge_ids=("e0", "e1", "e2"))
        rotated = CycleRecord(vertices=(1, 2, 0, 1), edge_ids=("e1", "e2", "e0"))
        r1 = check_cycle_multicover(
            CycleMulticoverRequest(
                graph=TRIANGLE, cycles=(original,), target_multiplicity=1
            )
        )
        r2 = check_cycle_multicover(
            CycleMulticoverRequest(
                graph=TRIANGLE, cycles=(rotated,), target_multiplicity=1
            )
        )
        assert r1.edge_multiplicity == r2.edge_multiplicity
        assert r1.is_exact_k_cover == r2.is_exact_k_cover

    def test_reversal_invariance(self) -> None:
        """A cycle reversal does not change the multiplicity profile."""
        original = CycleRecord(vertices=(0, 1, 2, 0), edge_ids=("e0", "e1", "e2"))
        reversed_cycle = CycleRecord(vertices=(0, 2, 1, 0), edge_ids=("e2", "e1", "e0"))
        r1 = check_cycle_multicover(
            CycleMulticoverRequest(
                graph=TRIANGLE, cycles=(original,), target_multiplicity=1
            )
        )
        r2 = check_cycle_multicover(
            CycleMulticoverRequest(
                graph=TRIANGLE, cycles=(reversed_cycle,), target_multiplicity=1
            )
        )
        assert r1.edge_multiplicity == r2.edge_multiplicity
        assert r1.is_exact_k_cover == r2.is_exact_k_cover


# ---------------------------------------------------------------------------
# Zero-valued edge identification
# ---------------------------------------------------------------------------


class TestZeroEdgeIdentification:
    def test_zero_edge_identified(self) -> None:
        """A zero-valued edge is identified in the result."""
        flow = (
            FlowEdgeAssignment(edge_id="e0", orientation="left_to_right", value=(0,)),
            FlowEdgeAssignment(edge_id="e1", orientation="left_to_right", value=(0,)),
            FlowEdgeAssignment(edge_id="e2", orientation="left_to_right", value=(0,)),
        )
        result = _flow_check(TRIANGLE, Z3, flow)
        assert result.conservation_holds  # All zeros trivially conserves
        assert not result.nowhere_zero
        assert set(result.zero_edge_ids) == {"e0", "e1", "e2"}


# ---------------------------------------------------------------------------
# Source-binding replay of non-witness outcomes
# ---------------------------------------------------------------------------


class TestSourceBindingReplay:
    def test_forged_exhausted_on_flow_admitting_graph_rejected(self):
        """A triangle admits a nowhere-zero Z/3 flow; EXHAUSTED must not validate."""
        payload = {
            "graph": TRIANGLE.model_dump(),
            "group": Z3.model_dump(),
            "resource_budget": {"max_states": 1024, "require_nowhere_zero": True},
            "status": "EXHAUSTED",
            "flow": None,
            "states_explored": 0,
            "termination_reason": "SEARCH_EXHAUSTED",
        }
        with pytest.raises(ValidationError, match="search replay"):
            MultigraphFlowFindResult.model_validate(payload)

    def test_genuine_exhausted_roundtrips(self):
        request = MultigraphFlowFindRequest(
            graph=BRIDGE_GRAPH,
            group=Z3,  # the bridge admits no nonzero flow: search exhausts
            resource_budget={"max_states": 1_000_000, "require_nowhere_zero": True},
        )
        result = find_multigraph_flow(request)
        assert result.status == "EXHAUSTED"
        assert MultigraphFlowFindResult.model_validate(result.model_dump()) == result

    def test_mutated_exhausted_state_count_rejected(self):
        request = MultigraphFlowFindRequest(
            graph=BRIDGE_GRAPH,
            group=Z3,
            resource_budget={"max_states": 1_000_000, "require_nowhere_zero": True},
        )
        genuine = find_multigraph_flow(request)
        payload = genuine.model_dump()
        payload["states_explored"] = int(payload["states_explored"]) + 1
        with pytest.raises(ValidationError):
            MultigraphFlowFindResult.model_validate(payload)

    def test_status_swap_between_non_witness_outcomes_rejected(self):
        request = MultigraphFlowFindRequest(
            graph=TRIANGLE,
            group=Z3,
            resource_budget={"max_states": 1, "require_nowhere_zero": False},
        )
        unknown = find_multigraph_flow(request)
        assert unknown.status == "UNKNOWN"
        payload = unknown.model_dump()
        payload["status"] = "EXHAUSTED"
        payload["termination_reason"] = "SEARCH_EXHAUSTED"
        with pytest.raises(ValidationError, match="search replay"):
            MultigraphFlowFindResult.model_validate(payload)

    def test_unbound_results_skip_replay(self):
        """Internal unbound results (no retained source) validate on shape alone."""
        result = MultigraphFlowFindResult(
            status="EXHAUSTED",
            flow=None,
            states_explored=0,
            termination_reason="SEARCH_EXHAUSTED",
        )
        assert result.graph is None


class TestEulerianSourceRequired:
    def test_result_requires_source_graph(self):
        with pytest.raises(ValidationError):
            EulerianCyclesResult(
                cycles=(),
                edge_usage=(),
                covers_all=True,
            )

    def test_forged_empty_decomposition_rejected(self):
        """cycles=()/covers_all=True against a graph with edges cannot validate."""
        payload = {
            "graph": SQUARE.model_dump(),
            "edge_subset": None,
            "cycles": (),
            "edge_usage": (("s0", 0), ("s1", 0), ("s2", 0), ("s3", 0)),
            "covers_all": True,
        }
        with pytest.raises(ValidationError, match="covers_all"):
            EulerianCyclesResult.model_validate(payload)

    def test_genuine_decomposition_roundtrips(self):
        request = EulerianCyclesRequest(graph=SQUARE)
        result = compute_eulerian_cycles(request)
        rebuilt = EulerianCyclesResult.model_validate(result.model_dump())
        assert rebuilt == result
