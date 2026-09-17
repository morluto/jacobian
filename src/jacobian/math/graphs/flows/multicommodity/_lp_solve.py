"""Exact rational-LP engine for multicommodity feasibility and congestion.

Arc-flow formulations over the maintained exact rational LP backend: one
nonnegative variable per commodity edge (plus a congestion ratio variable),
conservation equalities, and capacity inequalities.  Optimal vertices come
back as exact rationals and are verified independently: feasible tensors
replay through the witness checker, and Farkas balances replay in exact
arithmetic before any certificate is constructed.  Backend execution
failures stay operational and map to UNKNOWN, never to a verdict.
"""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.graphs.flows._models import FlowGraph
from jacobian.math.graphs.flows.multicommodity._models import (
    CommodityDemand,
    CommodityEdgeFlow,
    MulticommodityFlow,
    NetworkFarkasCertificate,
)
from jacobian.math.optimization._general_linear_program import general_linear_program
from jacobian.math.optimization._general_models import (
    MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS,
    MAX_GENERAL_LINEAR_PROGRAM_VARIABLES,
    MAX_GENERAL_RATIONAL_INPUT_DIGITS,
    GeneralFormRationalLinearProgram,
    GeneralRationalLinearProgramResult,
    RationalLinearConstraint,
    RationalLinearObjective,
    RationalLinearProgramVariable,
)

__all__ = [
    "LPExecutionExceededError",
    "admit_lp_envelope",
    "congestion_program",
    "feasibility_program",
    "solve_flow_program",
]


class LPExecutionExceededError(Exception):
    """The exact LP backend stopped within its operational envelope."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _rational(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def admit_lp_envelope(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
    extra_variables: int = 0,
) -> None:
    """Preflight the LP source size and input digits before formulation."""

    variables = len(commodities) * len(network.edges) + extra_variables
    constraints = len(commodities) * network.vertex_count + len(network.edges)
    if variables > MAX_GENERAL_LINEAR_PROGRAM_VARIABLES or (
        constraints > MAX_GENERAL_LINEAR_PROGRAM_CONSTRAINTS
    ):
        raise OperationResourceAdmissionError(
            location=("network", "commodities"),
            code="graph.multicommodity_lp_envelope",
            message=("the commodity-edge program exceeds the exact LP source envelope"),
        )
    for commodity in commodities:
        digits = canonical_rational_component_digits(commodity.demand)
        if digits > MAX_GENERAL_RATIONAL_INPUT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("commodities",),
                code="graph.multicommodity_demand_digits",
                message="a commodity demand exceeds the exact LP input digits",
            )
    for edge in network.edges:
        digits = canonical_rational_component_digits(edge.capacity)
        if digits > MAX_GENERAL_RATIONAL_INPUT_DIGITS:
            raise OperationResourceAdmissionError(
                location=("network",),
                code="graph.multicommodity_capacity_digits",
                message="an edge capacity exceeds the exact LP input digits",
            )


def _flow_variable_names(
    commodities: tuple[CommodityDemand, ...], edge_count: int
) -> list[str]:
    """Deterministic LP variable names in commodity, then edge, order."""

    return [
        f"f_{commodity_index}_{edge_index}"
        for commodity_index in range(len(commodities))
        for edge_index in range(edge_count)
    ]


def _conservation_rhs(commodity: CommodityDemand, vertex: int) -> CanonicalRational:
    if vertex == commodity.source:
        return commodity.demand
    if vertex == commodity.sink:
        return CanonicalRational.from_fraction(-commodity.demand.as_fraction())
    return _rational(0)


def feasibility_program(
    network: FlowGraph, commodities: tuple[CommodityDemand, ...]
) -> GeneralFormRationalLinearProgram:
    """Build min-0 arc-flow feasibility over one variable per commodity edge."""

    admit_lp_envelope(network, commodities)
    names = _flow_variable_names(commodities, len(network.edges))
    variables = tuple(
        RationalLinearProgramVariable(
            name=name, lower_bound=_rational(0), upper_bound=None
        )
        for name in names
    )
    zero = _rational(0)
    constraints: list[RationalLinearConstraint] = []
    for commodity_index, commodity in enumerate(commodities):
        base = commodity_index * len(network.edges)
        for vertex in range(network.vertex_count):
            row = [Fraction(0)] * len(names)
            for edge_index, edge in enumerate(network.edges):
                if edge.source == vertex:
                    row[base + edge_index] += 1
                if edge.target == vertex:
                    row[base + edge_index] -= 1
            constraints.append(
                RationalLinearConstraint(
                    label=f"cons:{commodity_index}:{vertex}",
                    coefficients=tuple(
                        CanonicalRational.from_fraction(value) for value in row
                    ),
                    relation="EQ",
                    rhs=_conservation_rhs(commodity, vertex),
                )
            )
    for edge_index, edge in enumerate(network.edges):
        row = [Fraction(0)] * len(names)
        for commodity_index in range(len(commodities)):
            row[commodity_index * len(network.edges) + edge_index] += 1
        constraints.append(
            RationalLinearConstraint(
                label=f"cap:{edge.source}>{edge.target}",
                coefficients=tuple(
                    CanonicalRational.from_fraction(value) for value in row
                ),
                relation="LE",
                rhs=edge.capacity,
            )
        )
    return GeneralFormRationalLinearProgram(
        variables=variables,
        objective=RationalLinearObjective(
            sense="MINIMIZE", coefficients=tuple(zero for _ in names)
        ),
        constraints=tuple(constraints),
    )


def congestion_program(
    network: FlowGraph, commodities: tuple[CommodityDemand, ...]
) -> GeneralFormRationalLinearProgram:
    """Build min-t congestion: loads bounded by t times capacities."""

    admit_lp_envelope(network, commodities, extra_variables=1)
    names = [*_flow_variable_names(commodities, len(network.edges)), "congestion_t"]
    time_index = len(names) - 1
    variables = tuple(
        RationalLinearProgramVariable(
            name=name, lower_bound=_rational(0), upper_bound=None
        )
        for name in names
    )
    zero = _rational(0)
    constraints: list[RationalLinearConstraint] = []
    for commodity_index, commodity in enumerate(commodities):
        base = commodity_index * len(network.edges)
        for vertex in range(network.vertex_count):
            row = [Fraction(0)] * len(names)
            for edge_index, edge in enumerate(network.edges):
                if edge.source == vertex:
                    row[base + edge_index] += 1
                if edge.target == vertex:
                    row[base + edge_index] -= 1
            constraints.append(
                RationalLinearConstraint(
                    label=f"cons:{commodity_index}:{vertex}",
                    coefficients=tuple(
                        CanonicalRational.from_fraction(value) for value in row
                    ),
                    relation="EQ",
                    rhs=_conservation_rhs(commodity, vertex),
                )
            )
    for edge_index, edge in enumerate(network.edges):
        row = [Fraction(0)] * len(names)
        for commodity_index in range(len(commodities)):
            row[commodity_index * len(network.edges) + edge_index] += 1
        row[time_index] -= edge.capacity.as_fraction()
        constraints.append(
            RationalLinearConstraint(
                label=f"cong:{edge.source}>{edge.target}",
                coefficients=tuple(
                    CanonicalRational.from_fraction(value) for value in row
                ),
                relation="LE",
                rhs=zero,
            )
        )
    objective = [Fraction(0)] * len(names)
    objective[time_index] = Fraction(1)
    return GeneralFormRationalLinearProgram(
        variables=variables,
        objective=RationalLinearObjective(
            sense="MINIMIZE",
            coefficients=tuple(
                CanonicalRational.from_fraction(value) for value in objective
            ),
        ),
        constraints=tuple(constraints),
    )


def solve_flow_program(
    program: GeneralFormRationalLinearProgram,
) -> GeneralRationalLinearProgramResult:
    """Run the exact backend, mapping operational failures to UNKNOWN data."""

    try:
        return general_linear_program(program)
    except (
        OperationResourceExhaustedError,
        OperationExecutionTimeoutError,
        OperationExecutionCancelledError,
    ) as exc:
        raise LPExecutionExceededError(str(exc)) from exc


def tensor_from_primal(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
    primal: tuple[CanonicalRational, ...],
) -> MulticommodityFlow:
    """Project an exact primal point onto the sparse flow tensor."""

    entries: list[CommodityEdgeFlow] = []
    for commodity_index, commodity in enumerate(commodities):
        base = commodity_index * len(network.edges)
        for edge_index, edge in enumerate(network.edges):
            amount = primal[base + edge_index]
            if amount.as_fraction() <= 0:
                continue
            entries.append(
                CommodityEdgeFlow(
                    commodity_id=commodity.commodity_id,
                    source=edge.source,
                    target=edge.target,
                    amount=amount,
                )
            )
    entries.sort(key=lambda entry: (entry.commodity_id, entry.source, entry.target))
    return MulticommodityFlow(
        network=network, commodities=commodities, entries=tuple(entries)
    )


def require_feasible_tensor(flow: MulticommodityFlow) -> None:
    """Replay a produced tensor through the witness checker, fail closed."""

    from jacobian.math.graphs.flows.multicommodity.operations import (
        check_multicommodity_flow_witness,
    )

    verdict = check_multicommodity_flow_witness(flow)
    if verdict.status != "FEASIBLE":
        raise RuntimeError("a produced flow tensor must verify feasible")


def require_congestion_tensor(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
    flow: MulticommodityFlow,
    congestion: Fraction,
) -> None:
    """Replay a congestion tensor: conservation, scaled caps, exact ratio."""

    from jacobian.math.graphs.flows.multicommodity.operations import (
        compute_multicommodity_flow_profile,
    )

    profile = compute_multicommodity_flow_profile(flow)
    if not profile.all_demands_routed:
        raise RuntimeError("a produced congestion tensor must route every demand")
    maximum = Fraction(0)
    for edge, edge_profile in zip(network.edges, profile.edge_profiles, strict=True):
        load = edge_profile.load.as_fraction()
        capacity = edge.capacity.as_fraction()
        if capacity == 0:
            if load != 0:
                raise RuntimeError(
                    "a produced congestion tensor loads no zero-capacity edge"
                )
            continue
        if load > congestion * capacity:
            raise RuntimeError(
                "every positive-capacity load must fit the congestion ratio"
            )
        ratio = load / capacity
        if ratio > maximum:
            maximum = ratio
    if maximum != congestion:
        raise RuntimeError("the congestion ratio must equal the maximum load ratio")


def network_farkas_certificate(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
    farkas_constraints: tuple[CanonicalRational, ...],
) -> NetworkFarkasCertificate:
    """Translate source Farkas multipliers to network coordinates and replay.

    Conservation rows come first in commodity, then vertex, order, followed
    by capacity rows in network edge order.  Potentials are free; prices are
    asserted nonnegative; every reduced stationarity is recomputed and the
    Farkas value asserted strictly negative before construction.
    """

    expected_rows = len(commodities) * network.vertex_count + len(network.edges)
    if len(farkas_constraints) != expected_rows:
        raise RuntimeError("Farkas multipliers must cover every source row")
    rows_per_commodity = network.vertex_count
    potentials: list[tuple[CanonicalRational, ...]] = []
    for commodity_index in range(len(commodities)):
        potentials.append(
            tuple(
                farkas_constraints[commodity_index * rows_per_commodity + vertex]
                for vertex in range(network.vertex_count)
            )
        )
    prices = tuple(farkas_constraints[len(commodities) * rows_per_commodity :])
    if len(prices) != len(network.edges):
        raise RuntimeError("Farkas multipliers must cover every capacity row")
    potential_values = [[value.as_fraction() for value in row] for row in potentials]
    price_values = [price.as_fraction() for price in prices]
    if any(price < 0 for price in price_values):
        raise RuntimeError("Farkas edge prices must be nonnegative")
    for commodity_index in range(len(commodities)):
        for edge_index, edge in enumerate(network.edges):
            slack = (
                price_values[edge_index]
                + potential_values[commodity_index][edge.source]
                - potential_values[commodity_index][edge.target]
            )
            if slack < 0:
                raise RuntimeError("every reduced stationarity must be nonnegative")
    value = Fraction(0)
    value += sum(
        commodity.demand.as_fraction()
        * (
            potential_values[commodity_index][commodity.source]
            - potential_values[commodity_index][commodity.sink]
        )
        for commodity_index, commodity in enumerate(commodities)
    )
    value += sum(
        price * edge.capacity.as_fraction()
        for price, edge in zip(price_values, network.edges, strict=True)
    )
    if value >= 0:
        raise RuntimeError("the Farkas value must be strictly negative")
    return NetworkFarkasCertificate(
        node_potentials=tuple(potentials),
        edge_prices=prices,
        farkas_value=CanonicalRational.from_fraction(value),
    )
