"""Canonical values and source-bound profiles for exact multicommodity flow."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Self

from pydantic import Field, PrivateAttr, StrictStr, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    parse_canonical_integer,
)
from jacobian.math.graphs.flow._models import FlowGraph

# This operation scans a sparse commodity-by-edge tensor and materializes one
# divergence value for every commodity/vertex pair.  Vertex count and edge
# count are owned by FlowGraph; admission controls the dense divergence table
# through the commodity-vertex cell budget and the whole returned value
# through the aggregate result envelope below.  Commodity, edge, and entry
# counts are never capped independently: entries are distinct commodity-by-
# edge cells and the kernel consumes sparse entries and per-edge sums rather
# than a dense tensor.
MAX_MULTICOMMODITY_EDGES = 512
MAX_COMMODITY_VERTEX_CELLS = 512

# Entry amounts and edge capacities are the only arithmetic operands, and each
# derived component is computed independently from only the operands that can
# reach it: a divergence cell sums the amounts incident to one
# commodity/vertex pair, an edge load sums the amounts carried by that edge, a
# slack subtracts one capacity from one such load, and the congestion ratio
# divides one load by one capacity.  Summation growth cannot be bounded from
# digit counts alone — shared denominators, cancellation, and lone operands
# all keep components far smaller than any worst case — so admission measures
# the actual reduced numerator and denominator of every completed component
# and enforces the canonical cap exactly there.  Fold work is bounded
# separately and BEFORE any cross-denominator arithmetic: the budget below is
# derived from what one fold of two canonical-bounded operands can produce,
# so no constructed fraction can outgrow it regardless of request size, and
# fold order cannot shrink the safe mathematical domain beyond that derived
# intermediate depth.  A request whose exact load equals its capacity
# therefore reports the exact zero slack and exactly-one congestion instead
# of phantom growth, while a request whose completed sums exceed the cap
# still fails closed.  Demands take part only in exact conservation
# comparisons, never in arithmetic, so they are covered by the measured
# source echo instead of these budgets.

# A result echoes its source tensor, then includes one divergence row per
# commodity-vertex cell, one edge row per network edge, and one congestion
# value. A derived rational occupies at most num+den+24 canonical JSON bytes
# when its own component bounds limit the two sides separately; the
# conservative row overhead reserves ASCII keys, labels, separators, and
# vertices. This envelope belongs to the profile operation, not to the
# canonical tensor value: request parsing performs the operation's single
# component scan, admits work and result envelope from it, and hands the
# measured components to the producer pass, while the replay pass always
# rescans independently -- so an accepted call executes exactly the two
# charged passes and the work ledger stays an exact per-call accounting.
MAX_PROFILE_RESULT_BYTES = 8 * 1024 * 1024
_DIVERGENCE_ROW_OVERHEAD_BYTES = 128
_EDGE_ROW_OVERHEAD_BYTES = 128
_RATIONAL_JSON_OVERHEAD_BYTES = 24
_PROFILE_RESULT_HEADER_BYTES = 1_024

# One pass performs at most 6F+E additions/subtractions, K negations, at
# most E divisions, and K*V+3E comparisons. Each sparse entry performs three
# bucketed numerator additions -- source divergence, sink divergence, and
# edge load -- and deposits its denominator into each of those three buckets,
# so the shared combination adds at most one fraction per distinct (bucket,
# denominator) pair: at most 3F denominator folds on top of the 3F numerator
# additions, plus one slack subtraction per edge. Sparse entries are distinct
# commodity-by-edge cells of the conceptual tensor, so F <= K*E <= 256 * 512.
# Every admitted commodity occupies V >= 2 distinct commodity-vertex cells
# because its source and sink differ, so the 512-cell divergence budget
# admits at most 256 commodities, one negation each, and FlowGraph's own
# 512-edge tuple bounds E. The shared derivative scan also classifies each
# edge's capacity with one zero-capacity comparison and checks each
# zero-capacity edge's load once: when a loaded zero-capacity edge exists the
# result's congestion is null, so no ratio is divided or compared at all;
# otherwise those checks pair every edge either with a zero-capacity load
# test or with one ratio division plus one running-maximum comparison, and
# the kernel's own edge loop needs only its load-vs-capacity feasibility
# test. With the admitted maxima this bounds each pass by 789,760 logical
# steps; producer plus exact result replay therefore costs at most 1,579,520.
MAX_SPARSE_FLOW_ENTRIES = (MAX_COMMODITY_VERTEX_CELLS // 2) * MAX_MULTICOMMODITY_EDGES
MAX_PROFILE_ADDITIONS_PER_PASS = 6 * MAX_SPARSE_FLOW_ENTRIES + MAX_MULTICOMMODITY_EDGES
MAX_PROFILE_NEGATIONS_PER_PASS = MAX_COMMODITY_VERTEX_CELLS // 2
MAX_PROFILE_DIVISIONS_PER_PASS = MAX_MULTICOMMODITY_EDGES
MAX_PROFILE_COMPARISONS_PER_PASS = (
    MAX_COMMODITY_VERTEX_CELLS + 3 * MAX_MULTICOMMODITY_EDGES
)
MAX_PROFILE_LOGICAL_STEPS = 2 * (
    MAX_PROFILE_ADDITIONS_PER_PASS
    + MAX_PROFILE_NEGATIONS_PER_PASS
    + MAX_PROFILE_DIVISIONS_PER_PASS
    + MAX_PROFILE_COMPARISONS_PER_PASS
)


class CommodityDemand(StrictModel):
    """One labelled source-to-sink demand in a directed capacitated network."""

    commodity_id: StrictStr = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
        description=(
            "Stable ASCII label for this commodity. Commodity tuples are sorted "
            "lexicographically by this field."
        ),
    )
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    demand: CanonicalRational

    @model_validator(mode="after")
    def require_distinct_positive_terminals(self) -> Self:
        if self.source == self.sink:
            raise PydanticCustomError(
                "graph.commodity_source_and_sink_must_be_distinct",
                "commodity source and sink must be distinct",
            )
        if self.demand.as_fraction() <= 0:
            raise PydanticCustomError(
                "graph.commodity_demand_must_be_strictly_positive",
                "commodity demand must be strictly positive",
            )
        return self


class CommodityEdgeFlow(StrictModel):
    """One positive sparse entry of a commodity-by-directed-edge flow tensor."""

    commodity_id: StrictStr = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
    )
    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    amount: CanonicalRational

    @model_validator(mode="after")
    def require_positive_amount(self) -> Self:
        if self.amount.as_fraction() <= 0:
            raise PydanticCustomError(
                "graph.sparse_flow_entries_must_have_strictly_positive_",
                "sparse flow entries must have strictly positive amounts",
            )
        return self


def _require_canonical_network(network: FlowGraph) -> tuple[tuple[int, int], ...]:
    edge_keys = tuple((edge.source, edge.target) for edge in network.edges)
    if edge_keys != tuple(sorted(edge_keys)):
        raise PydanticCustomError(
            "graph.network_edges_must_be_sorted_by_source_target",
            "network edges must be sorted by (source, target)",
        )
    return edge_keys


def _require_canonical_commodities(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
) -> tuple[str, ...]:
    commodity_ids = tuple(commodity.commodity_id for commodity in commodities)
    if commodity_ids != tuple(sorted(commodity_ids)):
        raise PydanticCustomError(
            "graph.commodities_must_be_sorted_by_commodity_id",
            "commodities must be sorted by commodity_id",
        )
    if len(set(commodity_ids)) != len(commodity_ids):
        raise PydanticCustomError(
            "graph.commodity_ids_must_be_unique", "commodity IDs must be unique"
        )
    for commodity in commodities:
        if not (
            commodity.source < network.vertex_count
            and commodity.sink < network.vertex_count
        ):
            raise PydanticCustomError(
                "graph.commodity_terminals_must_be_in_0_network_vertex_",
                "commodity terminals must be in 0..network.vertex_count-1",
            )
    if len(commodities) * network.vertex_count > MAX_COMMODITY_VERTEX_CELLS:
        raise PydanticCustomError(
            "graph.commodity_by_vertex_divergence_cell_count_exceeds",
            "commodity-by-vertex divergence cell count exceeds "
            f"{MAX_COMMODITY_VERTEX_CELLS}",
        )
    return commodity_ids


def _require_canonical_entries(
    entries: tuple[CommodityEdgeFlow, ...],
    *,
    edge_keys: tuple[tuple[int, int], ...],
    commodity_ids: tuple[str, ...],
) -> None:
    entry_keys = tuple(
        (entry.commodity_id, entry.source, entry.target) for entry in entries
    )
    if entry_keys != tuple(sorted(entry_keys)):
        raise PydanticCustomError(
            "graph.flow_entries_sorted_by_commodity_id_source",
            "flow entries must be sorted by (commodity_id, source, target)",
        )
    if len(set(entry_keys)) != len(entry_keys):
        raise PydanticCustomError(
            "graph.each_commodity_by_edge_flow_entry_may_occur_once",
            "each commodity-by-edge flow entry may occur once",
        )
    declared_edges = set(edge_keys)
    declared_commodities = set(commodity_ids)
    for entry in entries:
        if entry.commodity_id not in declared_commodities:
            raise PydanticCustomError(
                "graph.flow_entry_references_an_undeclared_commodity",
                "flow entry references an undeclared commodity",
            )
        if (entry.source, entry.target) not in declared_edges:
            raise PydanticCustomError(
                "graph.flow_entry_references_an_undeclared_directed_edg",
                "flow entry references an undeclared directed edge",
            )


def _component_sums(
    flow: MulticommodityFlow,
) -> tuple[dict[tuple[str, int], Fraction], dict[tuple[int, int], Fraction]]:
    """Return the exact divergence-cell and edge-load sums for one tensor."""

    divergences, loads, _fold_additions = _component_sums_with_folds(flow)
    return divergences, loads


def _component_sums_with_folds(
    flow: MulticommodityFlow,
) -> tuple[
    dict[tuple[str, int], Fraction],
    dict[tuple[int, int], Fraction],
    int,
]:
    """Return the exact sums plus how many fraction folds were performed.

    Amounts are bucketed by identical denominator and combined smallest
    denominator first: integer bucket sums never exceed the request volume,
    shared denominators add with zero growth, and every cross-denominator
    fold is pre-checked against the fold-intermediate budget from its
    operands' measured sides, so adversarial floods abort before
    constructing huge intermediate fractions. The canonical cap is enforced
    on the completed reduced sums by the component measurement, so fold
    order cannot shrink the safe mathematical domain. The fold counter
    increments inside the combination loop itself, so the profile work
    ledger charges exactly the additions this scan executes.
    """

    cell_buckets: dict[tuple[str, int], dict[int, int]] = {
        (commodity.commodity_id, vertex): {}
        for commodity in flow.commodities
        for vertex in range(flow.network.vertex_count)
    }
    edge_buckets: dict[tuple[int, int], dict[int, int]] = {
        (edge.source, edge.target): {} for edge in flow.network.edges
    }
    for entry in flow.entries:
        numerator = parse_canonical_integer(entry.amount.num)
        denominator = parse_canonical_integer(entry.amount.den)
        source_key = (entry.commodity_id, entry.source)
        target_key = (entry.commodity_id, entry.target)
        edge_key = (entry.source, entry.target)
        source_bucket = cell_buckets.setdefault(source_key, {})
        source_bucket[denominator] = source_bucket.get(denominator, 0) + numerator
        target_bucket = cell_buckets.setdefault(target_key, {})
        target_bucket[denominator] = target_bucket.get(denominator, 0) - numerator
        bucket = edge_buckets.setdefault(edge_key, {})
        bucket[denominator] = bucket.get(denominator, 0) + numerator

    fold_additions = 0

    def _combine(buckets: dict[int, int]) -> Fraction:
        nonlocal fold_additions
        total = Fraction(0)
        total_sides = (1, 1)
        for den in sorted(buckets, key=int.bit_length):
            operand = Fraction(buckets[den], den)
            num_digits, den_digits = _rational_side_bounds(operand)
            # The unreduced sum of a/b and c/d carries at most
            # len(b) + len(d) denominator digits and
            # max(len(a) + len(d), len(c) + len(b)) + 1 numerator digits,
            # so refusing folds on those predicted bounds enforces the
            # intermediate budget BEFORE any cross-denominator
            # multiplication, reduction, or decimal measurement runs:
            # no constructed fraction ever outgrows the budget-sized
            # envelope, whatever the request's total digit mass is.
            _require_side_within_fold_budget(
                max(total_sides[0] + den_digits, num_digits + total_sides[1]) + 1
            )
            _require_side_within_fold_budget(total_sides[1] + den_digits)
            total += operand
            fold_additions += 1
            total_sides = _rational_side_bounds(total)
        return total

    divergences = {key: _combine(bucket) for key, bucket in cell_buckets.items()}
    loads = {key: _combine(bucket) for key, bucket in edge_buckets.items()}
    return divergences, loads, fold_additions


# One cross-denominator fold adds two reduced rationals whose sides are
# measured before the arithmetic.  Every fold operand is itself bounded by
# the canonical cap -- bucket numerators sum only entry numerators sharing
# one denominator and grow by at most the logarithm of the entry count --
# so a fold whose accumulator sits inside the canonical cap has predicted
# unreduced sides of at most twice the cap plus one carry digit, exactly
# the composition of two maximal canonical operands.  The headroom below
# covers that carry, and the budget admits every such fold while any fold
# that would push the accumulator past it must first reduce back beneath
# the budget; monotone adversarial growth therefore aborts after
# constantly many cheap budget-sized folds instead of constructing
# multi-million-digit fractions.
MAX_PROFILE_FOLD_INTERMEDIATE_DIGITS = 2 * MAX_CANONICAL_RATIONAL_DIGITS + 8


def _require_side_within_fold_budget(digits: int) -> None:
    if digits > MAX_PROFILE_FOLD_INTERMEDIATE_DIGITS:
        raise PydanticCustomError(
            "graph.multicommodity_flow_profile_derives_digits_digit_fold",
            "multicommodity-flow profile derives a "
            f"{digits}-digit fold intermediate above the "
            f"{MAX_PROFILE_FOLD_INTERMEDIATE_DIGITS}"
            "-digit fold-intermediate budget",
        )


def _rational_side_bounds(value: Fraction) -> tuple[int, int]:
    return len(format_canonical_integer(abs(value.numerator))), len(
        format_canonical_integer(value.denominator)
    )


def _require_side_within_cap(digits: int) -> None:
    if digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise PydanticCustomError(
            "graph.multicommodity_flow_profile_derives_digits_digit_rational",
            "multicommodity-flow profile derives a "
            f"{digits}-digit rational above the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit canonical cap",
        )


def _measured_side_bounds(value: Fraction) -> tuple[int, int]:
    sides = _rational_side_bounds(value)
    _require_side_within_cap(sides[0])
    _require_side_within_cap(sides[1])
    return sides


def _edge_slacks_and_max_congestion(
    flow: MulticommodityFlow,
    loads: dict[tuple[int, int], Fraction],
) -> tuple[dict[tuple[int, int], Fraction], Fraction | None, int, int]:
    """Return each exact slack plus the congestion ratio when it is returned.

    Every slack is subtracted here exactly once. The same loop classifies
    each edge with one zero-capacity comparison and tests each zero-capacity
    edge's load once: a loaded zero-capacity edge forces the public result's
    ``congestion`` to be null, so in that case no positive-capacity ratio is
    divided, compared against a running maximum, capped, or priced -- the
    returned ratio is None and the division count is zero. Otherwise every
    positive-capacity edge contributes exactly one ratio division and one
    running-maximum comparison. The returned counts are the comparisons and
    divisions this scan executed, so the profile work ledger charges exactly
    the arithmetic this scan performs.
    """

    slacks: dict[tuple[int, int], Fraction] = {}
    positive_capacity_edges: list[tuple[Fraction, Fraction]] = []
    zero_capacity_violation = False
    comparisons = 0
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        capacity = edge.capacity.as_fraction()
        slacks[edge_key] = capacity - loads[edge_key]
        comparisons += 1
        if capacity == 0:
            comparisons += 1
            if loads[edge_key] > 0:
                zero_capacity_violation = True
        else:
            positive_capacity_edges.append((loads[edge_key], capacity))
    if zero_capacity_violation:
        return slacks, None, comparisons, 0
    max_ratio = Fraction(0)
    divisions = 0
    for load, capacity in positive_capacity_edges:
        candidate = load / capacity
        divisions += 1
        if candidate > max_ratio:
            max_ratio = candidate
        comparisons += 1
    return slacks, max_ratio, comparisons, divisions


def _measured_component_digit_bounds(
    divergences: dict[tuple[str, int], Fraction],
    loads: dict[tuple[int, int], Fraction],
    slacks: dict[tuple[int, int], Fraction],
    congestion_ratio: Fraction | None,
) -> tuple[
    dict[tuple[str, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    tuple[int, int] | None,
]:
    cell_bounds = {
        key: _measured_side_bounds(value) for key, value in divergences.items()
    }
    load_bounds = {key: _measured_side_bounds(value) for key, value in loads.items()}
    slack_bounds = {key: _measured_side_bounds(value) for key, value in slacks.items()}
    # A null congestion ratio is exactly the loaded zero-capacity-edge case in
    # which the result returns no congestion value: there is nothing to cap.
    congestion_bound = (
        None if congestion_ratio is None else _measured_side_bounds(congestion_ratio)
    )
    return cell_bounds, load_bounds, slack_bounds, congestion_bound


def _profile_component_digit_bounds(
    flow: MulticommodityFlow,
    component_sums: tuple[
        dict[tuple[str, int], Fraction],
        dict[tuple[int, int], Fraction],
    ]
    | None = None,
) -> tuple[
    dict[tuple[str, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    tuple[int, int] | None,
]:
    """Measure the exact numerator/denominator sizes of every derived row.

    Shared denominators, cancellation, and lone operands all keep real sums
    far below any digit-count worst case, so each component is measured as
    the exact reduced rational the kernel will produce; any accumulation that
    crosses the canonical cap aborts immediately instead of growing further.
    The congestion bound is None exactly when the result returns no
    congestion value.
    """

    divergences, loads = (
        component_sums if component_sums is not None else (_component_sums(flow))
    )
    slacks, congestion_ratio, _comparisons, _divisions = (
        _edge_slacks_and_max_congestion(flow, loads)
    )
    return _measured_component_digit_bounds(
        divergences, loads, slacks, congestion_ratio
    )


class AdmittedProfileScan(NamedTuple):
    """The once-computed components of one measured profile scan.

    Request parsing performs this scan, admits work and result envelope
    from it, and hands it to the producer pass, which otherwise recomputes
    it; the replay pass always rescans independently. Either way an
    accepted call executes exactly two arithmetic passes.

    ``congestion_ratio`` is None exactly when a loaded zero-capacity edge
    makes the public result's congestion null, so no ratio arithmetic ran;
    ``scan_comparisons`` and ``scan_divisions`` count the comparisons and
    divisions this scan executed for the exact work ledger.
    """

    divergences: dict[tuple[str, int], Fraction]
    loads: dict[tuple[int, int], Fraction]
    denominator_folds: int
    budget: int
    slacks: dict[tuple[int, int], Fraction]
    congestion_ratio: Fraction | None
    scan_divisions: int
    scan_comparisons: int
    cell_bounds: dict[tuple[str, int], tuple[int, int]]
    load_bounds: dict[tuple[int, int], tuple[int, int]]
    slack_bounds: dict[tuple[int, int], tuple[int, int]]
    congestion_bound: tuple[int, int] | None


def measured_profile_components(flow: MulticommodityFlow) -> AdmittedProfileScan:
    """Run the single bucketed component scan and derive every priced bound.

    The returned slacks and congestion ratio are the same measured components
    priced into the budget, so one producer or replay pass subtracts each
    slack exactly once and divides each positive-capacity ratio exactly once
    whenever the result returns a congestion value (and never divides one
    when a loaded zero-capacity edge forces it to be null); the work ledger
    charges that single execution. The per-row bound maps are those same
    measurements, letting envelope admission price the result without any
    second arithmetic pass.
    """

    divergences, loads, denominator_folds = _component_sums_with_folds(flow)
    slacks, congestion_ratio, scan_comparisons, scan_divisions = (
        _edge_slacks_and_max_congestion(flow, loads)
    )
    cell_bounds, load_bounds, slack_bounds, congestion_bound = (
        _measured_component_digit_bounds(divergences, loads, slacks, congestion_ratio)
    )
    component_sides = [
        *cell_bounds.values(),
        *load_bounds.values(),
        *slack_bounds.values(),
    ]
    if congestion_bound is not None:
        component_sides.append(congestion_bound)
    return AdmittedProfileScan(
        divergences=divergences,
        loads=loads,
        denominator_folds=denominator_folds,
        budget=max(max(sides) for sides in component_sides),
        slacks=slacks,
        congestion_ratio=congestion_ratio,
        scan_divisions=scan_divisions,
        scan_comparisons=scan_comparisons,
        cell_bounds=cell_bounds,
        load_bounds=load_bounds,
        slack_bounds=slack_bounds,
        congestion_bound=congestion_bound,
    )


def derived_profile_digit_budget(flow: MulticommodityFlow) -> int:
    """Return the exact digit bound shared by every derived profile component."""

    return measured_profile_components(flow).budget


def _require_profile_output_admission(flow: MulticommodityFlow) -> AdmittedProfileScan:
    """Admit the profile envelope and return its once-computed components.

    Request parsing runs this complete mathematical validation so every
    accepted ``math.run`` request reaches the kernel guaranteed admissible,
    and native callers get the same typed rejection before any result
    construction. The returned scan is reused as the producer pass, so an
    accepted call executes exactly the two charged passes.
    """

    _require_profile_source_room(flow)
    scan = measured_profile_components(flow)
    _require_admitted_profile_rows(
        flow,
        scan.cell_bounds,
        scan.load_bounds,
        scan.slack_bounds,
        scan.congestion_bound,
    )
    return scan


def _require_profile_source_room(flow: MulticommodityFlow) -> None:
    """Reject tensors whose echoed source leaves no room for any result.

    The echoed source is measured before any exact component work. Every
    admitted result contains at least the header, one congestion rational,
    and one divergence and one edge row (FlowGraph admits no zero-edge
    graphs and commodities at least one source/sink pair), so a serialized
    source that already leaves no room for that skeleton can never pass the
    priced estimate below and fails closed immediately instead of paying
    the full component scan first.
    """

    source_bytes = len(encode_strict_json(flow.model_dump(mode="json")))
    minimum_result_bytes = (
        source_bytes
        + _PROFILE_RESULT_HEADER_BYTES
        # One congestion rational and the mandatory first divergence row,
        # each priced at one digit per side like the estimate below.
        + _RATIONAL_JSON_OVERHEAD_BYTES
        + 2
        + _DIVERGENCE_ROW_OVERHEAD_BYTES
        + _RATIONAL_JSON_OVERHEAD_BYTES
        + 2
        # The mandatory first edge row prices two rationals.
        + _EDGE_ROW_OVERHEAD_BYTES
        + 2 * _RATIONAL_JSON_OVERHEAD_BYTES
        + 4
    )
    if minimum_result_bytes > MAX_PROFILE_RESULT_BYTES:
        raise PydanticCustomError(
            "graph.multicommodity_flow_profile_result_would_exceed_max",
            "multicommodity-flow profile result would exceed the "
            f"{MAX_PROFILE_RESULT_BYTES}-byte aggregate result bound",
        )


def _require_admitted_profile_rows(
    flow: MulticommodityFlow,
    cell_bounds: dict[tuple[str, int], tuple[int, int]],
    load_bounds: dict[tuple[int, int], tuple[int, int]],
    slack_bounds: dict[tuple[int, int], tuple[int, int]],
    congestion_bound: tuple[int, int] | None,
) -> None:
    """Price every returned row against the aggregate result envelope.

    The bounds are the ones the kernel's single measured scan already
    produced, so admission prices the result without a second arithmetic
    pass and the two-pass work ledger stays exact. A null congestion bound
    is the loaded zero-capacity-edge case whose result serializes a null:
    the reserved rational overhead alone covers that field's bytes.
    """

    divergence_bytes = sum(
        num_digits
        + den_digits
        + _RATIONAL_JSON_OVERHEAD_BYTES
        + _DIVERGENCE_ROW_OVERHEAD_BYTES
        for num_digits, den_digits in cell_bounds.values()
    )
    edge_bytes = 0
    congestion_bytes = (
        _RATIONAL_JSON_OVERHEAD_BYTES
        if congestion_bound is None
        else (sum(congestion_bound) + _RATIONAL_JSON_OVERHEAD_BYTES)
    )
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        load_num, load_den = load_bounds[edge_key]
        slack_num, slack_den = slack_bounds[edge_key]
        edge_bytes += (
            load_num
            + load_den
            + slack_num
            + slack_den
            + 2 * _RATIONAL_JSON_OVERHEAD_BYTES
            + _EDGE_ROW_OVERHEAD_BYTES
        )
    estimated_bytes = (
        len(encode_strict_json(flow.model_dump(mode="json")))
        + divergence_bytes
        + edge_bytes
        + congestion_bytes
        + _PROFILE_RESULT_HEADER_BYTES
    )
    if estimated_bytes > MAX_PROFILE_RESULT_BYTES:
        raise PydanticCustomError(
            "graph.multicommodity_flow_profile_result_would_exceed_max",
            "multicommodity-flow profile result would exceed the "
            f"{MAX_PROFILE_RESULT_BYTES}-byte aggregate result bound",
        )


class MulticommodityFlow(StrictModel):
    """A canonical sparse exact commodity-by-edge tensor over one FlowGraph.

    Every omitted tensor cell denotes exact zero.  Graph edges, commodities, and
    nonzero tensor entries are sorted so the value has one JSON representation.
    The source network and demand tuple remain attached to the tensor, allowing
    downstream operations to consume this value without reconstructing context.
    Only canonical representation bounds live here: each consumer operation
    enforces its own execution envelope at its own boundary.
    """

    network: FlowGraph = Field(
        description=(
            "Directed capacitated network. Its edges must be sorted by "
            "(source, target) for this canonical multicommodity-flow value."
        )
    )
    commodities: tuple[CommodityDemand, ...] = Field(
        min_length=1,
        description=(
            "Distinct commodity records sorted lexicographically by commodity_id. "
            "The count is bounded by the commodity-vertex and commodity-edge "
            "cell budgets rather than a fixed ceiling."
        ),
    )
    entries: tuple[CommodityEdgeFlow, ...] = Field(
        default=(),
        max_length=MAX_SPARSE_FLOW_ENTRIES,
        description=(
            "Nonzero tensor entries sorted by (commodity_id, source, target); "
            "omitted entries are exact zero."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_bounded_tensor(self) -> Self:
        edge_keys = _require_canonical_network(self.network)
        commodity_ids = _require_canonical_commodities(self.network, self.commodities)
        _require_canonical_entries(
            self.entries,
            edge_keys=edge_keys,
            commodity_ids=commodity_ids,
        )
        return self


class MulticommodityFlowProfileRequest(StrictModel):
    """Compute one exact conservation, load, slack, and congestion profile."""

    flow: MulticommodityFlow = Field(
        description=(
            "Canonical sparse tensor bounded by derived quantities: at most "
            "512 returned commodity-vertex cells (hence at most 256 "
            "commodities), at most the 512 network edges FlowGraph itself "
            "admits, nonzero entries bounded by the distinct commodity-by-edge "
            "cells they occupy, a per-component exact digit budget derived "
            "from each component's own operands, and an admitted aggregate "
            "result envelope below 8 MiB."
        )
    )

    _admitted_scan: AdmittedProfileScan | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def require_admitted_profile_work_and_result(self) -> Self:
        self._admitted_scan = _require_profile_output_admission(self.flow)
        return self


class CommodityDivergence(StrictModel):
    """The exact outgoing-minus-incoming flow for one commodity at one vertex."""

    commodity_id: StrictStr
    vertex: int = Field(ge=0, le=63)
    divergence: CanonicalRational


class EdgeLoadProfile(StrictModel):
    """Exact aggregate load and signed capacity slack for one directed edge."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    load: CanonicalRational
    slack: CanonicalRational


class MulticommodityFlowProfileWork(StrictModel):
    """Exact finite accounting for producer and source-bound result replay."""

    execution_passes_per_call: int = Field(ge=2, le=2)
    sparse_entries: int = Field(ge=0, le=MAX_SPARSE_FLOW_ENTRIES)
    commodity_vertex_cells: int = Field(ge=1, le=MAX_COMMODITY_VERTEX_CELLS)
    edge_cells: int = Field(ge=1, le=MAX_MULTICOMMODITY_EDGES)
    rational_additions_per_pass: int = Field(ge=0, le=MAX_PROFILE_ADDITIONS_PER_PASS)
    rational_negations_per_pass: int = Field(ge=0, le=MAX_PROFILE_NEGATIONS_PER_PASS)
    rational_divisions_per_pass: int = Field(ge=0, le=MAX_PROFILE_DIVISIONS_PER_PASS)
    exact_comparisons_per_pass: int = Field(ge=0, le=MAX_PROFILE_COMPARISONS_PER_PASS)
    logical_steps_per_call: int = Field(ge=0, le=MAX_PROFILE_LOGICAL_STEPS)


class MulticommodityFlowProfileResult(StrictModel):
    """A complete exact source-bound profile of a multicommodity flow tensor."""

    flow: MulticommodityFlow
    divergences: tuple[CommodityDivergence, ...] = Field(
        min_length=1,
        max_length=MAX_COMMODITY_VERTEX_CELLS,
        description=(
            "Dense rows sorted by (commodity_id, vertex), including exact zero "
            "divergences."
        ),
    )
    edge_profiles: tuple[EdgeLoadProfile, ...] = Field(
        min_length=1,
        max_length=MAX_MULTICOMMODITY_EDGES,
        description=(
            "One row per network edge, sorted by (source, target), including "
            "zero-load edges. Slack is capacity minus aggregate load."
        ),
    )
    all_demands_routed: bool
    capacity_feasible: bool
    congestion: CanonicalRational | None = Field(
        default=None,
        description=(
            "max(load/capacity) over positive-capacity edges, or null exactly "
            "when a zero-capacity edge carries positive load."
        ),
    )
    work: MulticommodityFlowProfileWork

    @model_validator(mode="after")
    def require_exact_source_bound_profile(self) -> Self:
        from jacobian.math.graphs.multicommodity_flow._kernel import profile_components

        expected = profile_components(self.flow)
        actual = (
            self.divergences,
            self.edge_profiles,
            self.all_demands_routed,
            self.capacity_feasible,
            self.congestion,
            self.work,
        )
        if actual != expected:
            raise PydanticCustomError(
                "graph.result_must_match_the_exact_multicommodity_flow_",
                "result must match the exact multicommodity-flow profile",
            )
        return self


__all__ = [
    "MAX_COMMODITY_VERTEX_CELLS",
    "MAX_MULTICOMMODITY_EDGES",
    "MAX_PROFILE_ADDITIONS_PER_PASS",
    "MAX_PROFILE_COMPARISONS_PER_PASS",
    "MAX_PROFILE_DIVISIONS_PER_PASS",
    "MAX_PROFILE_FOLD_INTERMEDIATE_DIGITS",
    "MAX_PROFILE_LOGICAL_STEPS",
    "MAX_PROFILE_NEGATIONS_PER_PASS",
    "MAX_PROFILE_RESULT_BYTES",
    "MAX_SPARSE_FLOW_ENTRIES",
    "CommodityDemand",
    "CommodityDivergence",
    "CommodityEdgeFlow",
    "EdgeLoadProfile",
    "MulticommodityFlow",
    "MulticommodityFlowProfileRequest",
    "MulticommodityFlowProfileResult",
    "MulticommodityFlowProfileWork",
    "derived_profile_digit_budget",
]
