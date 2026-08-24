"""Pure exact kernels for bounded multicommodity-flow profiles."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.math.graphs.multicommodity_flow._models import (
    MAX_PROFILE_RATIONAL_DIGITS,
    CommodityDivergence,
    EdgeLoadProfile,
    MulticommodityFlow,
    MulticommodityFlowProfileWork,
)


def _wire(value: Fraction) -> CanonicalRational:
    result = CanonicalRational.from_fraction(value)
    require_bounded_rational(
        result,
        max_digits=MAX_PROFILE_RATIONAL_DIGITS,
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
            divergence=_wire(divergences[(commodity_id, vertex)]),
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
    all_demands_routed = all(
        divergences[key] == expected for key, expected in expected_divergences.items()
    )

    edge_rows: list[EdgeLoadProfile] = []
    capacity_feasible = True
    congestion: Fraction | None = Fraction(0)
    positive_capacity_edges = 0
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        capacity = edge.capacity.as_fraction()
        load = loads[edge_key]
        if load > capacity:
            capacity_feasible = False
        if capacity == 0:
            if load > 0:
                congestion = None
        else:
            positive_capacity_edges += 1
            ratio = load / capacity
            if congestion is not None and ratio > congestion:
                congestion = ratio
        edge_rows.append(
            EdgeLoadProfile(
                source=edge.source,
                target=edge.target,
                load=_wire(load),
                slack=_wire(capacity - load),
            )
        )

    sparse_entries = len(flow.entries)
    divergence_cells = len(divergence_rows)
    edge_cells = len(edge_rows)
    # Each sparse entry adds to source divergence, sink divergence, and edge
    # load. Each edge then subtracts its load from capacity. One demand is
    # negated per commodity for its sink target. Every edge compares load to
    # capacity and capacity to zero, then compares either load to zero or its
    # ratio to the running congestion maximum.
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
        None if congestion is None else _wire(congestion),
        work,
    )


__all__ = ["profile_components"]
