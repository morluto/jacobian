"""Pure exact kernels for bounded multicommodity-flow profiles."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.math.graphs.flows.multicommodity._models import (
    AdmittedProfileScan,
    CommodityDivergence,
    EdgeLoadProfile,
    MulticommodityFlow,
    MulticommodityFlowProfileWork,
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
    admitted: AdmittedProfileScan,
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
    and aggregate edge loads.  The aggregate result envelope is admitted from
    the same measured components -- the echoed source before the scan, every
    priced row afterwards -- so admission adds no arithmetic pass of its own.
    The caller supplies the admitted scan, which serves as the producer pass
    verbatim. Request and result parsing perform structural checks only and
    never re-enter this kernel.
    """

    scan = admitted

    commodity_ids = tuple(commodity.commodity_id for commodity in flow.commodities)
    divergences = scan.divergences
    loads = scan.loads
    denominator_folds = scan.denominator_folds
    budget = scan.budget
    slacks = scan.slacks
    congestion_ratio = scan.congestion_ratio
    scan_divisions = scan.scan_divisions
    scan_comparisons = scan.scan_comparisons

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
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        load = loads[edge_key]
        if load > edge.capacity.as_fraction():
            capacity_feasible = False
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
    # once by the shared derivative scan that returned the budget above; that
    # scan also owns the zero-capacity classification, so when a loaded
    # zero-capacity edge makes the congestion value null it divides and
    # compares no ratios at all, and otherwise each positive-capacity ratio
    # is divided exactly once and compared against the running maximum once.
    # Its reported comparison and division counts charge exactly the work it
    # executed, and this loop adds only its load-vs-capacity feasibility test
    # per edge. One demand is negated per commodity for its sink target, and
    # the demand scan above compares all commodity/vertex cells rather than
    # short-circuiting at the first mismatch, so every charged comparison is
    # executed exactly once.
    additions = 3 * sparse_entries + denominator_folds + edge_cells
    negations = len(flow.commodities)
    divisions = scan_divisions
    comparisons = divergence_cells + edge_cells + scan_comparisons
    per_pass = additions + negations + divisions + comparisons
    work = MulticommodityFlowProfileWork(
        execution_passes_per_call=1,
        sparse_entries=sparse_entries,
        commodity_vertex_cells=divergence_cells,
        edge_cells=edge_cells,
        rational_additions_per_pass=additions,
        rational_negations_per_pass=negations,
        rational_divisions_per_pass=divisions,
        exact_comparisons_per_pass=comparisons,
        logical_steps_per_call=per_pass,
    )
    return (
        divergence_rows,
        tuple(edge_rows),
        all_demands_routed,
        capacity_feasible,
        None
        if congestion_ratio is None
        else _wire(congestion_ratio, max_digits=budget),
        work,
    )


__all__ = ["profile_components"]
