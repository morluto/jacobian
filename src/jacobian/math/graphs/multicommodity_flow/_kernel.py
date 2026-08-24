"""Pure exact kernels for bounded multicommodity-flow profiles."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.math.graphs.multicommodity_flow._models import (
    CommodityDivergence,
    EdgeLoadProfile,
    MulticommodityFlow,
    MulticommodityFlowProfileWork,
    _component_sums_with_folds,
    derived_profile_components_from_sums,
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
    divergences, loads, denominator_folds = _component_sums_with_folds(flow)
    budget, slacks, max_congestion_ratio = derived_profile_components_from_sums(
        flow, divergences, loads
    )

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
        edge_rows.append(
            EdgeLoadProfile(
                source=edge.source,
                target=edge.target,
                load=_wire(load, max_digits=budget),
                slack=_wire(slacks[edge_key], max_digits=budget),
            )
        )

    sparse_entries = len(flow.entries)
    divergence_cells = len(divergence_rows)
    edge_cells = len(edge_rows)
    # Each sparse entry performs three bucketed integer additions -- source
    # divergence, sink divergence, and edge load -- and deposits its
    # denominator into each of those three buckets. The shared sum scan then
    # folds every distinct bucket denominator exactly once, counted inside
    # the fold loop itself, so ``denominator_folds`` equals the fraction
    # additions the scan executed. Each edge's slack is subtracted exactly
    # once by the shared scan that returned the budget above, which also
    # divides one ratio per positive-capacity edge; this loop reuses those
    # measured components. One demand is negated per commodity for its sink
    # target. Every edge is compared four times -- load against capacity and
    # capacity against zero in this loop, capacity against zero in the shared
    # scan, then either load against zero or its ratio against the running
    # congestion maximum. The demand scan above compares all commodity/vertex
    # cells rather than short-circuiting at the first mismatch, so every
    # charged comparison is executed exactly once.
    additions = 3 * sparse_entries + denominator_folds + edge_cells
    negations = len(flow.commodities)
    divisions = positive_capacity_edges
    comparisons = divergence_cells + 4 * edge_cells
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
