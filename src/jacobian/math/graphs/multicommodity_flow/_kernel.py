"""Pure exact kernels for bounded multicommodity-flow profiles."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.math.graphs.multicommodity_flow._models import (
    CommodityDivergence,
    EdgeLoadProfile,
    MulticommodityFlow,
    MulticommodityFlowProfileWork,
    derived_profile_digit_budget,
)


def _wire(value: Fraction, *, max_digits: int) -> CanonicalRational:
    result = CanonicalRational.from_fraction(value)
    require_bounded_rational(
        result,
        max_digits=max_digits,
        label="derived multicommodity-flow profile rational",
    )
    return result


def profile_components(
    flow: MulticommodityFlow,
) -> tuple[
    tuple[CommodityDivergence, ...],
    tuple[EdgeLoadProfile, ...],
    bool,
    bool,
    CanonicalRational | None,
    MulticommodityFlowProfileWork,
]:
    """Return the complete deterministic profile components for ``flow``.

    One pass initializes and updates dense commodity/vertex divergence cells
    and aggregate edge loads.  Result-model validation calls this same pure
    kernel once more, so the returned work ledger charges both passes.
    """

    commodity_ids = tuple(commodity.commodity_id for commodity in flow.commodities)
    edge_keys = tuple((edge.source, edge.target) for edge in flow.network.edges)
    budget = derived_profile_digit_budget(flow)
    divergences = {
        (commodity_id, vertex): Fraction(0)
        for commodity_id in commodity_ids
        for vertex in range(flow.network.vertex_count)
    }
    loads = {edge_key: Fraction(0) for edge_key in edge_keys}
    for entry in flow.entries:
        amount = entry.amount.as_fraction()
        divergences[(entry.commodity_id, entry.source)] += amount
        divergences[(entry.commodity_id, entry.target)] -= amount
        loads[(entry.source, entry.target)] += amount

    divergence_rows = tuple(
        CommodityDivergence(
            commodity_id=commodity_id,
            vertex=vertex,
            divergence=_wire(divergences[(commodity_id, vertex)], max_digits=budget),
        )
        for commodity_id in commodity_ids
        for vertex in range(flow.network.vertex_count)
    )
    expected_divergences = {
        (commodity.commodity_id, vertex): (
            demand
            if vertex == commodity.source
            else -demand
            if vertex == commodity.sink
            else Fraction(0)
        )
        for commodity in flow.commodities
        for demand in (commodity.demand.as_fraction(),)
        for vertex in range(flow.network.vertex_count)
    }
    mismatched_divergence_cells = sum(
        divergences[key] != expected for key, expected in expected_divergences.items()
    )
    all_demands_routed = mismatched_divergence_cells == 0

    edge_rows: list[EdgeLoadProfile] = []
    capacity_feasible = True
    max_congestion_ratio = Fraction(0)
    zero_capacity_violation = False
    positive_capacity_edges = 0
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        capacity = edge.capacity.as_fraction()
        load = loads[edge_key]
        if load > capacity:
            capacity_feasible = False
        if capacity == 0:
            if load > 0:
                zero_capacity_violation = True
        else:
            positive_capacity_edges += 1
            ratio = load / capacity
            if ratio > max_congestion_ratio:
                max_congestion_ratio = ratio
        edge_rows.append(
            EdgeLoadProfile(
                source=edge.source,
                target=edge.target,
                load=_wire(load, max_digits=budget),
                slack=_wire(capacity - load, max_digits=budget),
            )
        )

    sparse_entries = len(flow.entries)
    divergence_cells = len(divergence_rows)
    edge_cells = len(edge_rows)
    # Each sparse entry adds to source divergence, sink divergence, and edge
    # load. Each edge then subtracts its load from capacity. One demand is
    # negated per commodity for its sink target. Every edge compares load to
    # capacity and capacity to zero, then compares either load to zero or its
    # ratio to the running congestion maximum. The demand scan above compares
    # all commodity/vertex cells rather than short-circuiting at the first
    # mismatch, so every charged comparison is executed exactly once.
    additions = 3 * sparse_entries + edge_cells
    negations = len(flow.commodities)
    divisions = positive_capacity_edges
    comparisons = divergence_cells + 3 * edge_cells
    per_pass = additions + negations + divisions + comparisons
    work = MulticommodityFlowProfileWork(
        execution_passes_per_call=2,
        sparse_entries=sparse_entries,
        commodity_vertex_cells=divergence_cells,
        edge_cells=edge_cells,
        rational_additions_per_pass=additions,
        rational_negations_per_pass=negations,
        rational_divisions_per_pass=divisions,
        exact_comparisons_per_pass=comparisons,
        logical_steps_per_call=2 * per_pass,
    )
    return (
        divergence_rows,
        tuple(edge_rows),
        all_demands_routed,
        capacity_feasible,
        None
        if zero_capacity_violation
        else _wire(max_congestion_ratio, max_digits=budget),
        work,
    )


__all__ = ["profile_components"]
