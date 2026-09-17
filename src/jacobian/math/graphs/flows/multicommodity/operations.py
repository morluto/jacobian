"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise, product

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.flows._models import FlowGraph
from jacobian.math.graphs.flows.multicommodity._kernel import profile_components
from jacobian.math.graphs.flows.multicommodity._lp_solve import (
    LPExecutionExceededError,
    admit_lp_envelope,
    congestion_program,
    feasibility_program,
    network_farkas_certificate,
    require_congestion_tensor,
    require_feasible_tensor,
    solve_flow_program,
    tensor_from_primal,
)
from jacobian.math.graphs.flows.multicommodity._models import (
    AdmittedProfileScan,
    CommodityDemand,
    CommodityVertexViolation,
    EdgeCapacityViolation,
    MinimumCongestionResult,
    MulticommodityFeasibilityResult,
    MulticommodityFlow,
    MulticommodityFlowProfileResult,
    MulticommodityFlowWitnessCheckResult,
    UnsplittablePath,
    UnsplittablePathViolation,
    UnsplittableRouting,
    UnsplittableRoutingCheckResult,
    UnsplittableRoutingFindResult,
    _require_canonical_commodities,
    _require_canonical_network,
    _require_profile_output_admission,
)


def compute_multicommodity_flow_profile(
    flow: MulticommodityFlow,
) -> MulticommodityFlowProfileResult:
    """Compute the exact bounded load and conservation profile of one tensor.

    The canonical tensor value carries only representation bounds; this
    execution boundary performs the profile's semantic admission and exact
    computation. Request parsing remains structural.
    """

    return _profile_result(flow, _require_profile_output_admission(flow))


def _profile_result(
    flow: MulticommodityFlow,
    admitted: AdmittedProfileScan,
) -> MulticommodityFlowProfileResult:
    (
        divergences,
        edge_profiles,
        all_demands_routed,
        capacity_feasible,
        congestion,
        work,
    ) = profile_components(flow, admitted)
    return MulticommodityFlowProfileResult._from_kernel(
        flow,
        divergences=divergences,
        edge_profiles=edge_profiles,
        all_demands_routed=all_demands_routed,
        capacity_feasible=capacity_feasible,
        congestion=congestion,
        work=work,
    )


def check_multicommodity_flow_witness(
    flow: MulticommodityFlow,
) -> MulticommodityFlowWitnessCheckResult:
    """Exactly replay one submitted tensor and return its feasibility verdict.

    The admitted profile kernel computes the commodity-vertex divergences and
    aggregate edge loads once, in exact rational arithmetic.  This adapter then
    classifies those already-exact components against the declared terminals
    and capacities; it introduces no new arithmetic beyond exact comparisons
    and never consults a floating-point solver.
    """

    profile = compute_multicommodity_flow_profile(flow)
    expected: dict[tuple[str, int], Fraction] = {}
    for commodity in flow.commodities:
        demand = commodity.demand.as_fraction()
        for vertex in range(flow.network.vertex_count):
            if vertex == commodity.source:
                expected[(commodity.commodity_id, vertex)] = demand
            elif vertex == commodity.sink:
                expected[(commodity.commodity_id, vertex)] = -demand
            else:
                expected[(commodity.commodity_id, vertex)] = Fraction(0)

    violating_commodities: set[str] = set()
    violating_vertices: list[CommodityVertexViolation] = []
    for row in profile.divergences:
        target = expected[(row.commodity_id, row.vertex)]
        if row.divergence.as_fraction() != target:
            violating_commodities.add(row.commodity_id)
            violating_vertices.append(
                CommodityVertexViolation(
                    commodity_id=row.commodity_id,
                    vertex=row.vertex,
                    divergence=row.divergence,
                    expected=CanonicalRational.from_fraction(target),
                )
            )

    violating_edges: list[EdgeCapacityViolation] = []
    for edge, edge_profile in zip(
        flow.network.edges, profile.edge_profiles, strict=True
    ):
        if edge_profile.load.as_fraction() > edge.capacity.as_fraction():
            violating_edges.append(
                EdgeCapacityViolation(
                    source=edge.source,
                    target=edge.target,
                    load=edge_profile.load,
                    capacity=edge.capacity,
                )
            )

    return MulticommodityFlowWitnessCheckResult._from_kernel(
        flow,
        divergences=profile.divergences,
        edge_profiles=profile.edge_profiles,
        all_demands_routed=profile.all_demands_routed,
        capacity_feasible=profile.capacity_feasible,
        congestion=profile.congestion,
        violating_commodities=tuple(sorted(violating_commodities)),
        violating_vertices=tuple(violating_vertices),
        violating_edges=tuple(violating_edges),
        work=profile.work,
    )


def _admit_flow_contract(
    network: FlowGraph, commodities: tuple[CommodityDemand, ...]
) -> None:
    """Require the canonical contract shared by the solve-side operations."""

    try:
        _require_canonical_network(network)
        _require_canonical_commodities(network, commodities)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error
    if not commodities:
        raise OperationDomainValidationError(
            location=("commodities",),
            code="graph.multicommodity_contract_needs_a_commodity",
            message="a flow contract needs at least one commodity",
        )


def solve_multicommodity_feasibility(
    network: FlowGraph, commodities: tuple[CommodityDemand, ...]
) -> MulticommodityFeasibilityResult:
    """Solve exact rational multicommodity-flow feasibility.

    The arc-flow program runs on the maintained exact rational LP backend;
    no floating-point solve participates.  FEASIBLE carries an exact tensor
    verified by the witness checker; INFEASIBLE carries a network Farkas
    certificate whose balance replays in exact arithmetic; UNKNOWN carries
    the bounded backend reason.  A truncated computation never becomes a
    verdict.
    """

    _admit_flow_contract(network, commodities)
    program = feasibility_program(network, commodities)
    try:
        outcome = solve_flow_program(program)
    except LPExecutionExceededError as exc:
        return MulticommodityFeasibilityResult._from_kernel(
            network,
            commodities,
            "UNKNOWN",
            unknown_reason="LP_EXECUTION_BOUND_EXCEEDED",
            unknown_detail=exc.detail,
        )
    if outcome.status == "OPTIMAL":
        if outcome.primal_candidate is None:
            raise RuntimeError("an optimal LP outcome carries its primal point")
        flow = tensor_from_primal(network, commodities, outcome.primal_candidate)
        require_feasible_tensor(flow)
        return MulticommodityFeasibilityResult._from_kernel(
            network, commodities, "FEASIBLE", flow=flow
        )
    if outcome.status == "INFEASIBLE":
        if outcome.farkas_constraints is None:
            raise RuntimeError("an infeasible LP outcome carries its Farkas data")
        certificate = network_farkas_certificate(
            network, commodities, outcome.farkas_constraints
        )
        return MulticommodityFeasibilityResult._from_kernel(
            network, commodities, "INFEASIBLE", certificate=certificate
        )
    raise RuntimeError("a feasibility program is never unbounded")


def verify_multicommodity_feasibility(
    claim: MulticommodityFeasibilityResult,
) -> bool:
    """Check a feasibility claim by re-solving within the same envelope."""
    try:
        return (
            solve_multicommodity_feasibility(claim.network, claim.commodities) == claim
        )
    except OperationDomainValidationError:
        return False


def solve_minimum_congestion(
    network: FlowGraph, commodities: tuple[CommodityDemand, ...]
) -> MinimumCongestionResult:
    """Minimize exact congestion over rational multicommodity flows.

    The min-t arc-flow program runs on the maintained exact rational LP
    backend.  FEASIBLE carries the exact minimum ratio with an attaining
    tensor; INFEASIBLE carries a network Farkas certificate for the
    underlying routability; UNKNOWN carries the bounded backend reason.
    """

    _admit_flow_contract(network, commodities)
    program = congestion_program(network, commodities)
    try:
        outcome = solve_flow_program(program)
    except LPExecutionExceededError as exc:
        return MinimumCongestionResult._from_kernel(
            network,
            commodities,
            "UNKNOWN",
            unknown_reason="LP_EXECUTION_BOUND_EXCEEDED",
            unknown_detail=exc.detail,
        )
    if outcome.status == "OPTIMAL":
        if outcome.primal_candidate is None or outcome.primal_objective is None:
            raise RuntimeError("an optimal LP outcome carries its primal point")
        congestion = outcome.primal_objective.as_fraction()
        flow = tensor_from_primal(network, commodities, outcome.primal_candidate)
        require_congestion_tensor(network, commodities, flow, congestion)
        return MinimumCongestionResult._from_kernel(
            network,
            commodities,
            "FEASIBLE",
            congestion=outcome.primal_objective,
            flow=flow,
        )
    if outcome.status == "INFEASIBLE":
        if outcome.farkas_constraints is None:
            raise RuntimeError("an infeasible LP outcome carries its Farkas data")
        certificate = network_farkas_certificate(
            network, commodities, outcome.farkas_constraints
        )
        return MinimumCongestionResult._from_kernel(
            network, commodities, "INFEASIBLE", certificate=certificate
        )
    raise RuntimeError("a congestion program is never unbounded")


def verify_minimum_congestion(claim: MinimumCongestionResult) -> bool:
    """Check a congestion claim by re-solving within the same envelope."""
    try:
        return solve_minimum_congestion(claim.network, claim.commodities) == claim
    except OperationDomainValidationError:
        return False


def check_unsplittable_routing(
    routing: UnsplittableRouting,
) -> UnsplittableRoutingCheckResult:
    """Check one submitted one-path-per-commodity routing exactly."""

    network = routing.network
    capacities = {
        (edge.source, edge.target): edge.capacity.as_fraction()
        for edge in network.edges
    }
    edge_set = set(capacities)
    demands = {
        commodity.commodity_id: (
            commodity.source,
            commodity.sink,
            commodity.demand.as_fraction(),
        )
        for commodity in routing.commodities
    }
    path_violations: list[UnsplittablePathViolation] = []
    loads: dict[tuple[int, int], Fraction] = {}
    for path in routing.paths:
        source, sink, demand = demands[path.commodity_id]
        vertices = path.vertices
        if vertices[0] != source or vertices[-1] != sink:
            path_violations.append(
                UnsplittablePathViolation(
                    commodity_id=path.commodity_id,
                    detail=("path endpoints must be the commodity source and sink"),
                )
            )
            continue
        steps = [(first, second) for first, second in pairwise(vertices)]
        if any(step not in edge_set for step in steps):
            path_violations.append(
                UnsplittablePathViolation(
                    commodity_id=path.commodity_id,
                    detail="every path step must be a network edge",
                )
            )
            continue
        for step in steps:
            loads[step] = loads.get(step, Fraction(0)) + demand
    edge_violations = [
        violation
        for edge in network.edges
        for violation in _capacity_violation(
            edge.source,
            edge.target,
            loads.get((edge.source, edge.target), Fraction(0)),
            edge.capacity.as_fraction(),
        )
    ]
    if path_violations or edge_violations:
        return UnsplittableRoutingCheckResult._from_kernel(
            routing,
            status="INFEASIBLE",
            path_violations=tuple(path_violations),
            edge_violations=tuple(edge_violations),
        )
    congestion: CanonicalRational | None = None
    peak = Fraction(0)
    for (source, target), load in loads.items():
        capacity = capacities[(source, target)]
        if capacity == 0:
            continue
        ratio = load / capacity
        if ratio > peak:
            peak = ratio
    if loads:
        congestion = CanonicalRational.from_fraction(peak)
    return UnsplittableRoutingCheckResult._from_kernel(
        routing, status="FEASIBLE", congestion=congestion
    )


def _capacity_violation(
    source: int, target: int, load: Fraction, capacity: Fraction
) -> tuple[EdgeCapacityViolation, ...]:
    if load <= capacity:
        return ()
    return (
        EdgeCapacityViolation(
            source=source,
            target=target,
            load=CanonicalRational.from_fraction(load),
            capacity=CanonicalRational.from_fraction(capacity),
        ),
    )


def _simple_paths(
    adjacency: dict[int, list[int]],
    source: int,
    sink: int,
    cap: int,
) -> tuple[list[tuple[int, ...]], bool]:
    """Enumerate simple source-to-sink paths in deterministic order.

    Returns the inventoried paths with a completeness flag that is false
    exactly when more than ``cap`` paths exist (one extra path is explored
    to decide completeness).
    """

    paths: list[tuple[int, ...]] = []
    stopped = False

    def visit(vertex: int, trail: tuple[int, ...]) -> None:
        nonlocal stopped
        if stopped:
            return
        if vertex == sink:
            paths.append(trail)
            if len(paths) > cap:
                stopped = True
            return
        for target in adjacency.get(vertex, ()):
            if target in trail:
                continue
            visit(target, (*trail, target))

    visit(source, (source,))
    return paths[:cap], not stopped


def _inventory_product(counts: tuple[int, ...]) -> int:
    """Multiply per-commodity path inventories exactly."""

    total = 1
    for count in counts:
        total *= count
    return total


def find_unsplittable_routing(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
    max_paths_per_commodity: int = 256,
    combination_budget: int = 50000,
) -> UnsplittableRoutingFindResult:
    """Find one feasible one-path-per-commodity routing by bounded search.

    Simple source-to-sink paths enumerate per commodity in deterministic
    adjacency order up to ``max_paths_per_commodity``; combinations enumerate
    row-major up to ``combination_budget`` with each combination decided by
    the exact routing checker.  FOUND carries the first feasible routing
    with its certificate; EXHAUSTED carries the exact combination receipt;
    UNKNOWN carries the bounded stop reason.  A truncated search never
    yields a negative conclusion.
    """

    _admit_flow_contract(network, commodities)
    if (
        type(max_paths_per_commodity) is not int
        or not 1 <= max_paths_per_commodity <= 4096
    ):
        raise OperationDomainValidationError(
            location=("max_paths_per_commodity",),
            code="graph.routing_path_cap_range",
            message="the per-commodity path cap stays within its envelope",
        )
    if type(combination_budget) is not int or not 1 <= combination_budget <= 50000:
        raise OperationDomainValidationError(
            location=("combination_budget",),
            code="graph.routing_combination_budget_range",
            message="the combination budget stays within its envelope",
        )
    admit_lp_envelope(network, commodities)
    adjacency: dict[int, list[int]] = {}
    for edge in network.edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    for targets in adjacency.values():
        targets.sort()
    inventories: list[list[tuple[int, ...]]] = []
    capped: int | None = None
    for position, commodity in enumerate(commodities):
        paths, complete = _simple_paths(
            adjacency, commodity.source, commodity.sink, max_paths_per_commodity
        )
        inventories.append(paths)
        if not complete and capped is None:
            capped = position
    counts = tuple(len(paths) for paths in inventories)
    if capped is not None:
        return UnsplittableRoutingFindResult._from_kernel(
            network,
            commodities,
            max_paths_per_commodity,
            combination_budget,
            "UNKNOWN",
            paths_per_commodity=counts,
            total_combinations=_inventory_product(counts),
            current_commodity=commodities[capped].commodity_id,
            stop_reason="PATH_CAP_EXCEEDED",
        )
    total = _inventory_product(counts)
    examined = 0
    for combination in product(*(range(count) for count in counts)):
        if examined >= combination_budget:
            return UnsplittableRoutingFindResult._from_kernel(
                network,
                commodities,
                max_paths_per_commodity,
                combination_budget,
                "UNKNOWN",
                paths_per_commodity=counts,
                combinations_examined=examined,
                total_combinations=total,
                stop_reason="COMBINATION_BUDGET_EXCEEDED",
            )
        routing = UnsplittableRouting(
            network=network,
            commodities=commodities,
            paths=tuple(
                UnsplittablePath(
                    commodity_id=commodity.commodity_id,
                    vertices=inventories[position][choice],
                )
                for position, (commodity, choice) in enumerate(
                    zip(commodities, combination, strict=True)
                )
            ),
        )
        certificate = check_unsplittable_routing(routing)
        examined += 1
        if certificate.status == "FEASIBLE":
            return UnsplittableRoutingFindResult._from_kernel(
                network,
                commodities,
                max_paths_per_commodity,
                combination_budget,
                "FOUND",
                routing=routing,
                certificate=certificate,
                paths_per_commodity=tuple(len(paths) for paths in inventories),
                combinations_examined=examined,
                total_combinations=total,
            )
    return UnsplittableRoutingFindResult._from_kernel(
        network,
        commodities,
        max_paths_per_commodity,
        combination_budget,
        "EXHAUSTED",
        paths_per_commodity=tuple(len(paths) for paths in inventories),
        combinations_examined=examined,
        total_combinations=total,
    )


def verify_unsplittable_routing_check(
    claim: UnsplittableRoutingCheckResult,
) -> bool:
    """Check a routing verdict by replaying its paths and loads."""
    try:
        return check_unsplittable_routing(claim.routing) == claim
    except (OperationDomainValidationError, ValueError):
        return False


def verify_unsplittable_routing_find(
    claim: UnsplittableRoutingFindResult,
) -> bool:
    """Check a routing search by replaying it within its bounds."""
    try:
        return (
            find_unsplittable_routing(
                claim.network,
                claim.commodities,
                claim.max_paths_per_commodity,
                claim.combination_budget,
            )
            == claim
        )
    except (OperationDomainValidationError, ValueError):
        return False


__all__ = [
    "check_multicommodity_flow_witness",
    "check_unsplittable_routing",
    "compute_multicommodity_flow_profile",
    "find_unsplittable_routing",
    "solve_minimum_congestion",
    "solve_multicommodity_feasibility",
    "verify_minimum_congestion",
    "verify_multicommodity_feasibility",
    "verify_unsplittable_routing_check",
    "verify_unsplittable_routing_find",
]
