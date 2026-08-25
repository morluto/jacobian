"""Domain-owned finite-multigraph flow and cycle operations.

These operations implement exact checking and bounded search for finite-
Abelian-group-valued flows on oriented loopless multigraphs, deterministic
Eulerian cycle decomposition, and cycle-multicover verification.  NetworkX is
used only as a private traversal backend for the Eulerian decomposition.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx

from jacobian.math.graphs.multigraph._models import (
    CycleMulticoverRequest,
    CycleMulticoverResult,
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
    MultigraphFlowSearchBudget,
    VertexDivergence,
    _FlowSearchOutcome,
)
from jacobian.math.graphs.multigraph._orientation import oriented_endpoints

__all__ = [
    "check_cycle_multicover",
    "check_multigraph_flow",
    "compute_eulerian_cycles",
    "find_multigraph_flow",
]


# ---------------------------------------------------------------------------
# Flow check
# ---------------------------------------------------------------------------


def check_multigraph_flow(
    request: MultigraphFlowCheckRequest,
) -> MultigraphFlowCheckResult:
    """Check a finite-Abelian flow by recomputing every vertex sum exactly.

    For each vertex, the signed divergence is the sum of outgoing flow values
    minus the sum of incoming flow values (componentwise modular arithmetic).
    Conservation holds when every vertex divergence is the zero element.
    """
    graph = request.graph
    group = request.group

    # Build per-vertex signed sums.  Outgoing edges contribute ``+value``;
    # incoming edges contribute ``-value`` (i.e. ``group.negate(value)``).
    vertex_out: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    vertex_in: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    vertex_incident: dict[int, set[str]] = {v: set() for v in range(graph.vertex_count)}

    for assign in request.edge_values:
        edge = graph.edge_by_id(assign.edge_id)
        tail, head = oriented_endpoints(edge, assign.orientation)
        value = group.normalize(assign.value)
        vertex_out[tail].append(value)
        vertex_in[head].append(value)
        vertex_incident[tail].add(assign.edge_id)
        vertex_incident[head].add(assign.edge_id)

    divergence_ledger: list[VertexDivergence] = []
    conservation_holds = True
    for vertex in range(graph.vertex_count):
        out_sum = group.sum(tuple(vertex_out[vertex]))
        in_sum = group.sum(tuple(vertex_in[vertex]))
        # divergence = out_sum - in_sum = out_sum + negate(in_sum)
        divergence = group.add(out_sum, group.negate(in_sum))
        holds = group.is_zero(divergence)
        if not holds:
            conservation_holds = False
        incident = sorted(vertex_incident[vertex])
        divergence_ledger.append(
            VertexDivergence(
                vertex=vertex,
                coordinates=divergence,
                incident_edge_ids=tuple(incident),
                conservation_holds=holds,
            )
        )

    # Identify zero-valued edges
    zero_edge_ids = [
        assign.edge_id for assign in request.edge_values if group.is_zero(assign.value)
    ]
    zero_edge_ids.sort()
    nowhere_zero = len(zero_edge_ids) == 0

    return MultigraphFlowCheckResult(
        graph=graph,
        group=group,
        edge_flow_records=request.edge_values,
        divergence_ledger=tuple(divergence_ledger),
        zero_edge_ids=tuple(zero_edge_ids),
        nowhere_zero=nowhere_zero,
        conservation_holds=conservation_holds,
    )


# ---------------------------------------------------------------------------
# Bounded flow search
# ---------------------------------------------------------------------------


def _enumerate_group_elements(group: FiniteAbelianGroup) -> list[tuple[int, ...]]:
    """Enumerate all group elements in lexicographic order."""
    moduli = group.moduli
    elements: list[tuple[int, ...]] = []
    _enumerate_recursive(moduli, 0, (), elements)
    return elements


def _enumerate_recursive(
    moduli: tuple[int, ...],
    index: int,
    current: tuple[int, ...],
    accumulator: list[tuple[int, ...]],
) -> None:
    if index == len(moduli):
        accumulator.append(current)
        return
    for value in range(moduli[index]):
        _enumerate_recursive(moduli, index + 1, (*current, value), accumulator)


def _check_flow_conservation(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    assignments: list[FlowEdgeAssignment],
) -> bool:
    """Return True when the given flow assignments satisfy conservation."""
    vertex_out: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    vertex_in: dict[int, list[tuple[int, ...]]] = {
        v: [] for v in range(graph.vertex_count)
    }
    for assign in assignments:
        edge = graph.edge_by_id(assign.edge_id)
        tail, head = oriented_endpoints(edge, assign.orientation)
        value = group.normalize(assign.value)
        vertex_out[tail].append(value)
        vertex_in[head].append(value)
    for vertex in range(graph.vertex_count):
        out_sum = group.sum(tuple(vertex_out[vertex]))
        in_sum = group.sum(tuple(vertex_in[vertex]))
        if not group.is_zero(group.add(out_sum, group.negate(in_sum))):
            return False
    return True


def _build_edge_choices(
    group: FiniteAbelianGroup,
    require_nz: bool,
) -> list[tuple[Literal["left_to_right", "right_to_left"], tuple[int, ...]]]:
    """Build the per-edge (orientation, value) choice list."""
    candidate_values = _enumerate_group_elements(group)
    if require_nz:
        candidate_values = [v for v in candidate_values if not group.is_zero(v)]
    orientations: tuple[Literal["left_to_right", "right_to_left"], ...] = (
        "left_to_right",
        "right_to_left",
    )
    choices: list[
        tuple[Literal["left_to_right", "right_to_left"], tuple[int, ...]]
    ] = []
    for orientation in orientations:
        for value in candidate_values:
            choices.append((orientation, value))
    return choices


def _incremental_balances_hold(net: list[list[int]], moduli: tuple[int, ...]) -> bool:
    """Return True when every signed component imbalance is zero in the group."""
    return all(
        component % moduli[k] == 0
        for vertex_balance in net
        for k, component in enumerate(vertex_balance)
    )


def _leaf_found_outcome(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    net: list[list[int]],
    moduli: tuple[int, ...],
    assignments: list[FlowEdgeAssignment],
    states_explored: int,
) -> _FlowSearchOutcome | None:
    """Evaluate one complete assignment; return the FOUND outcome or None.

    The incremental balances decide candidacy in O(vertices * dimension);
    the authoritative conservation replay then confirms the witness.
    """
    if not _incremental_balances_hold(net, moduli):
        return None
    if not _check_flow_conservation(graph, group, assignments):
        return None
    return _FlowSearchOutcome(
        status="FOUND",
        flow=tuple(assignments),
        states_explored=states_explored,
        termination_reason="WITNESS_FOUND",
    )


def _balance_apply(
    net: list[list[int]],
    edge: MultigraphEdge,
    assignment: FlowEdgeAssignment,
    sign: int,
) -> None:
    """Add (sign=1) or remove (sign=-1) one assignment's divergence update."""
    tail, head = oriented_endpoints(edge, assignment.orientation)
    for k, component in enumerate(assignment.value):
        net[tail][k] += sign * component
        net[head][k] -= sign * component


def _pop_balanced_assignment(
    assignments: list[FlowEdgeAssignment],
    edges: tuple[MultigraphEdge, ...],
    net: list[list[int]],
) -> None:
    """Pop the deepest assignment and undo its divergence update."""
    assignment = assignments.pop()
    _balance_apply(net, edges[len(assignments)], assignment, -1)


def _search_dfs(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    choices_per_edge: list[
        list[tuple[Literal["left_to_right", "right_to_left"], tuple[int, ...]]]
    ],
    max_states: int,
) -> _FlowSearchOutcome:
    """Run the bounded DFS search and return the unbound outcome."""
    num_edges = len(graph.edges)
    edges = graph.edges
    moduli = group.moduli
    dimension = len(moduli)
    states_explored = 0
    # Incremental divergence bookkeeping: net[v][k] accumulates the signed
    # k-component flow imbalance at vertex v over currently assigned edges.
    # Pushing or popping an assignment updates two vertices in O(dimension),
    # so leaf conservation costs O(vertices * dimension) instead of
    # rescanning every edge by ID (quadratic in the edge count) per leaf;
    # per-state work stays uniformly bounded by the charged state count.
    net: list[list[int]] = [[0] * dimension for _ in range(graph.vertex_count)]

    # Iterative DFS that expands one branch at a time and charges the
    # budget on every partial state. This avoids eagerly pushing all
    # children while copying prefix lists, which would retain up to
    # ``branching * depth`` assignments even when ``max_states == 1``.
    assignments: list[FlowEdgeAssignment] = []
    next_index: list[int] = [0]

    while next_index:
        depth = len(assignments)
        if depth == num_edges:
            # Complete assignment was already counted when the final edge
            # was pushed; its incremental balances were maintained on the
            # way down, so evaluating conservation is O(vertices * dimension).
            found = _leaf_found_outcome(
                graph, group, net, moduli, assignments, states_explored
            )
            if found is not None:
                return found
            # Backtrack from leaf: pop placeholder and last assignment.
            next_index.pop()
            if not next_index:
                break
            _pop_balanced_assignment(assignments, edges, net)
            continue

        choices = choices_per_edge[depth]
        idx = next_index[depth]
        if idx >= len(choices):
            # Exhausted all choices at this depth — backtrack.
            next_index.pop()
            if assignments:
                _pop_balanced_assignment(assignments, edges, net)
            continue

        # Advance pointer for this depth so the next sibling is tried
        # after backtracking.
        next_index[depth] = idx + 1

        # Charge the budget for the new partial state before materialising it.
        states_explored += 1
        if states_explored > max_states:
            return _FlowSearchOutcome(
                status="UNKNOWN",
                flow=None,
                states_explored=states_explored - 1,
                termination_reason="STATE_BUDGET_EXCEEDED",
            )

        orientation, value = choices[idx]
        edge = edges[depth]
        new_assign = FlowEdgeAssignment(
            edge_id=edge.edge_id,
            orientation=orientation,
            value=value,
        )
        assignments.append(new_assign)
        _balance_apply(net, edge, new_assign, 1)
        next_index.append(0)

    return _FlowSearchOutcome(
        status="EXHAUSTED",
        flow=None,
        states_explored=states_explored,
        termination_reason="SEARCH_EXHAUSTED",
    )


def _search_flow_unbound(
    graph: LooplessMultigraph,
    group: FiniteAbelianGroup,
    resource_budget: MultigraphFlowSearchBudget,
) -> _FlowSearchOutcome:
    """Run one bounded flow search and return it without a bound source.

    The returned result carries no ``graph``/``group``/``resource_budget``
    binding, so source-binding validators can replay it without recursion.
    """

    require_nz = resource_budget.require_nowhere_zero

    # Special case: a graph with no edges trivially has the empty flow.
    if not graph.edges:
        return _FlowSearchOutcome(
            status="FOUND",
            flow=(),
            states_explored=0,
            termination_reason="SPECIAL_CASE",
        )

    # Per-edge candidate values: all group elements (or nonzero if required).
    choices = _build_edge_choices(group, require_nz)
    if not choices:
        return _FlowSearchOutcome(
            status="EXHAUSTED",
            flow=None,
            states_explored=0,
            termination_reason="SEARCH_EXHAUSTED",
        )

    choices_per_edge = [choices] * len(graph.edges)
    return _search_dfs(graph, group, choices_per_edge, resource_budget.max_states)


def find_multigraph_flow(
    request: MultigraphFlowFindRequest,
) -> MultigraphFlowFindResult:
    """Search for a finite-Abelian flow on a loopless multigraph.

    The search enumerates all possible per-edge group-element assignments and
    all orientations.  When ``require_nowhere_zero`` is true, the zero element
    is excluded from the per-edge domain.  The search is bounded by
    ``resource_budget.max_states``; if the budget is exceeded before the
    complete search space is covered, the result is ``UNKNOWN``.  If the
    complete space is covered with no witness, the result is ``EXHAUSTED``.
    """

    inner = _search_flow_unbound(request.graph, request.group, request.resource_budget)
    return MultigraphFlowFindResult(
        graph=request.graph,
        group=request.group,
        resource_budget=request.resource_budget,
        status=inner.status,
        flow=inner.flow,
        states_explored=inner.states_explored,
        termination_reason=inner.termination_reason,
    )


# ---------------------------------------------------------------------------
# Eulerian cycle decomposition
# ---------------------------------------------------------------------------


def _build_multigraph_nx(
    graph: LooplessMultigraph, edge_ids: list[str]
) -> nx.MultiGraph[int]:
    """Build a NetworkX MultiGraph from the declared edge IDs."""
    g: nx.MultiGraph[int] = nx.MultiGraph()
    g.add_nodes_from(range(graph.vertex_count))
    for eid in edge_ids:
        edge = graph.edge_by_id(eid)
        g.add_edge(edge.left, edge.right, key=eid)
    return g


def _decompose_circuit_into_simple_cycles(
    circuit: list[tuple[int, int, str]],
) -> list[CycleRecord]:
    """Decompose an Eulerian circuit into simple cycles.

    Walks the circuit and extracts simple cycles greedily: whenever a vertex
    is revisited, the sub-walk since its first occurrence forms a simple cycle.
    """
    cycles: list[CycleRecord] = []
    # Build the vertex and edge sequence from the circuit
    if not circuit:
        return cycles

    # The circuit is a list of (u, v, key) edges.  Build vertex sequence.
    vertices: list[int] = [circuit[0][0]]
    edge_ids: list[str] = []
    for u, v, key in circuit:
        # The circuit edges may have either orientation; we follow the
        # walk so the "current" vertex should match u or v.
        if vertices[-1] == u:
            vertices.append(v)
        else:
            vertices.append(u)
        edge_ids.append(key)

    # Now extract simple cycles from the closed walk
    remaining_vertices = list(vertices)
    remaining_edge_ids = list(edge_ids)

    while remaining_edge_ids:
        cycle_verts: list[int] = []
        cycle_edges: list[str] = []
        seen: dict[int, int] = {}

        for i, (v, _eid) in enumerate(
            zip(remaining_vertices, remaining_edge_ids, strict=False)
        ):
            if v in seen:
                # Found a cycle from seen[v] to i
                start_idx = seen[v]
                cycle_verts = remaining_vertices[start_idx : i + 1]
                cycle_edges = remaining_edge_ids[start_idx:i]
                # Remove the cycle from the remaining walk
                remaining_vertices = (
                    remaining_vertices[:start_idx] + remaining_vertices[i:]
                )
                remaining_edge_ids = (
                    remaining_edge_ids[:start_idx] + remaining_edge_ids[i:]
                )
                break
            seen[v] = i
        else:
            # No cycle found — the remaining walk is itself a simple cycle
            if remaining_edge_ids:
                cycle_verts = list(remaining_vertices)
                cycle_edges = list(remaining_edge_ids)
                remaining_vertices = []
                remaining_edge_ids = []

        if cycle_edges:
            cycles.append(
                CycleRecord(
                    vertices=tuple(cycle_verts),
                    edge_ids=tuple(cycle_edges),
                )
            )

    return cycles


def compute_eulerian_cycles(
    request: EulerianCyclesRequest,
) -> EulerianCyclesResult:
    """Decompose an edge multiset into edge-disjoint cycles.

    Uses NetworkX's Eulerian circuit algorithm on a MultiGraph.  Any induced
    degree parity is accepted: if the edge set is not Eulerian (some vertex
    has odd degree), the result is an empty decomposition with
    ``covers_all=False``.
    """
    graph = request.graph
    if request.edge_subset is not None:
        edge_ids = list(request.edge_subset)
    else:
        edge_ids = [edge.edge_id for edge in graph.edges]

    if not edge_ids:
        return EulerianCyclesResult(
            graph=graph,
            edge_subset=request.edge_subset,
            cycles=(),
            edge_usage=(),
            covers_all=True,
        )

    # Check Eulerian condition: every vertex has even degree in the edge subset
    degree: dict[int, int] = dict.fromkeys(range(graph.vertex_count), 0)
    for eid in edge_ids:
        edge = graph.edge_by_id(eid)
        degree[edge.left] += 1
        degree[edge.right] += 1
    if any(d % 2 != 0 for d in degree.values()):
        return EulerianCyclesResult(
            graph=graph,
            edge_subset=request.edge_subset,
            cycles=(),
            edge_usage=tuple((eid, 0) for eid in sorted(edge_ids)),
            covers_all=False,
        )

    # Build NetworkX MultiGraph
    g = _build_multigraph_nx(graph, edge_ids)

    # Handle disconnected graphs: decompose each connected component
    all_cycles: list[CycleRecord] = []
    edge_usage: dict[str, int] = dict.fromkeys(edge_ids, 0)

    non_isolated = [v for v in g.nodes() if g.degree(v) > 0]
    if non_isolated:
        subgraph = g.subgraph(non_isolated)
        for component in sorted(nx.connected_components(subgraph)):
            comp_subgraph = subgraph.subgraph(component)
            if comp_subgraph.number_of_edges() == 0:
                continue
            start = min(comp_subgraph.nodes())
            circuit = list(nx.eulerian_circuit(comp_subgraph, source=start, keys=True))
            cycles = _decompose_circuit_into_simple_cycles(circuit)
            for cycle in cycles:
                all_cycles.append(cycle)
                for eid in cycle.edge_ids:
                    edge_usage[eid] += 1

    usage_tuple = tuple((eid, edge_usage[eid]) for eid in sorted(edge_usage))
    covers_all = all(edge_usage[eid] == 1 for eid in edge_ids)

    return EulerianCyclesResult(
        graph=graph,
        edge_subset=request.edge_subset,
        cycles=tuple(all_cycles),
        edge_usage=usage_tuple,
        covers_all=covers_all,
    )


# ---------------------------------------------------------------------------
# Cycle multicover check
# ---------------------------------------------------------------------------


def check_cycle_multicover(
    request: CycleMulticoverRequest,
) -> CycleMulticoverResult:
    """Check that a cycle family covers each edge exactly ``k`` times.

    Each cycle is validated against graph incidence: consecutive vertices
    must be connected by the declared edge ID, and every edge ID must exist
    in the graph.  Cycles may appear in any ordering, rotation, or reversal.
    The operation scores per-edge multiplicity.
    """
    graph = request.graph
    k = request.target_multiplicity

    edge_multiplicity: dict[str, int] = {edge.edge_id: 0 for edge in graph.edges}
    cycle_validity: list[bool] = []

    for cycle in request.cycles:
        cycle_valid = True
        for i, eid in enumerate(cycle.edge_ids):
            edge_valid = True
            if eid not in edge_multiplicity:
                edge_valid = False
                cycle_valid = False
            else:
                v_from = cycle.vertices[i]
                v_to = cycle.vertices[i + 1]
                edge = graph.edge_by_id(eid)
                if not (
                    (edge.left == v_from and edge.right == v_to)
                    or (edge.right == v_from and edge.left == v_to)
                ):
                    edge_valid = False
                    cycle_valid = False
            if edge_valid:
                edge_multiplicity[eid] += 1
        cycle_validity.append(cycle_valid)

    missing = sorted(eid for eid, count in edge_multiplicity.items() if count < k)
    overcovered = sorted(eid for eid, count in edge_multiplicity.items() if count > k)
    all_valid = all(cycle_validity)
    is_exact = all_valid and not missing and not overcovered

    multiplicity_tuple = tuple(
        (eid, edge_multiplicity[eid]) for eid in sorted(edge_multiplicity)
    )

    return CycleMulticoverResult(
        graph=graph,
        cycles=request.cycles,
        target_multiplicity=k,
        cycle_validity=tuple(cycle_validity),
        edge_multiplicity=multiplicity_tuple,
        missing_edge_ids=tuple(missing),
        overcovered_edge_ids=tuple(overcovered),
        is_exact_k_cover=is_exact,
    )
