"""Canonical values and source-bound profiles for exact multicommodity flow."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictStr, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
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
# divides one load by one capacity.  Numerator and denominator growth are
# tracked separately: a reduced sum denominator divides the operand lcm and
# so accumulates each operand's denominator digits once; a reduced sum
# numerator adds at most its largest operand numerator on top of that common
# denominator plus eight carry digits for up to 256 accumulated endpoints;
# the slack subtraction pushes the numerator through both denominators once
# and adds one carry digit; and the division crosses the load with the
# capacity numerator only.  A component fed by a single operand equals that
# already-canonical operand exactly, so its own two sides are its bounds and
# no summation growth is charged; when that lone amount also equals the edge
# capacity, the slack is the exact zero rational and the congestion ratio is
# exactly one.  An edge carrying no amount has the exact zero load, so only
# single-digit zero operands compose with its capacity.  A component stays
# admissible when each side of this split bound fits the canonical cap on
# its own.  Demands take part only in exact conservation comparisons, never
# in arithmetic, so they are covered by the measured source echo instead of
# these budgets.
_DERIVED_DIGIT_SLACK = 8

# A result echoes its source tensor, then includes one divergence row per
# commodity-vertex cell, one edge row per network edge, and one congestion
# value. A derived rational occupies at most num+den+24 canonical JSON bytes
# when its own component bounds limit the two sides separately; the
# conservative row overhead reserves ASCII keys, labels, separators, and
# vertices. Admission measures the echoed source exactly, prices exact-zero
# loads as the single-digit zero rational, and prices every other row from
# its own numerator and denominator bounds against this aggregate envelope,
# keeping the serialized result inside the envelope with headroom under the
# 10 MiB transport limit.
MAX_PROFILE_RESULT_BYTES = 8 * 1024 * 1024
_DIVERGENCE_ROW_OVERHEAD_BYTES = 128
_EDGE_ROW_OVERHEAD_BYTES = 128
_RATIONAL_JSON_OVERHEAD_BYTES = 24
_PROFILE_RESULT_HEADER_BYTES = 1_024

# One pass performs at most 3F+E additions/subtractions, K negations, E
# divisions, and K*V+3E comparisons. Sparse entries are distinct
# commodity-by-edge cells of the conceptual tensor, so F <= K*E <= 256 * 512.
# Every admitted commodity occupies V >= 2 distinct commodity-vertex cells
# because its source and sink differ, so the 512-cell divergence budget
# admits at most 256 commodities, one negation each, and FlowGraph's own
# 512-edge tuple bounds E. With the admitted maxima this is 396,544 logical
# steps per pass; producer plus exact result replay therefore costs at most
# 793,088.
MAX_PROFILE_LOGICAL_STEPS = 793_088
MAX_PROFILE_ADDITIONS_PER_PASS = 393_728
MAX_PROFILE_NEGATIONS_PER_PASS = MAX_COMMODITY_VERTEX_CELLS // 2
MAX_PROFILE_DIVISIONS_PER_PASS = 512
MAX_PROFILE_COMPARISONS_PER_PASS = 2_048
MAX_SPARSE_FLOW_ENTRIES = (MAX_COMMODITY_VERTEX_CELLS // 2) * MAX_MULTICOMMODITY_EDGES


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
            raise ValueError("commodity source and sink must be distinct")
        if self.demand.as_fraction() <= 0:
            raise ValueError("commodity demand must be strictly positive")
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
            raise ValueError("sparse flow entries must have strictly positive amounts")
        return self


def _require_canonical_network(network: FlowGraph) -> tuple[tuple[int, int], ...]:
    edge_keys = tuple((edge.source, edge.target) for edge in network.edges)
    if edge_keys != tuple(sorted(edge_keys)):
        raise ValueError("network edges must be sorted by (source, target)")
    return edge_keys


def _require_canonical_commodities(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
) -> tuple[str, ...]:
    commodity_ids = tuple(commodity.commodity_id for commodity in commodities)
    if commodity_ids != tuple(sorted(commodity_ids)):
        raise ValueError("commodities must be sorted by commodity_id")
    if len(set(commodity_ids)) != len(commodity_ids):
        raise ValueError("commodity IDs must be unique")
    for commodity in commodities:
        if not (
            commodity.source < network.vertex_count
            and commodity.sink < network.vertex_count
        ):
            raise ValueError("commodity terminals must be in 0..network.vertex_count-1")
    if len(commodities) * network.vertex_count > MAX_COMMODITY_VERTEX_CELLS:
        raise ValueError(
            "commodity-by-vertex divergence cell count exceeds "
            f"{MAX_COMMODITY_VERTEX_CELLS}"
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
        raise ValueError(
            "flow entries must be sorted by (commodity_id, source, target)"
        )
    if len(set(entry_keys)) != len(entry_keys):
        raise ValueError("each commodity-by-edge flow entry may occur once")
    declared_edges = set(edge_keys)
    declared_commodities = set(commodity_ids)
    for entry in entries:
        if entry.commodity_id not in declared_commodities:
            raise ValueError("flow entry references an undeclared commodity")
        if (entry.source, entry.target) not in declared_edges:
            raise ValueError("flow entry references an undeclared directed edge")


def _profile_component_digit_bounds(
    flow: MulticommodityFlow,
) -> tuple[
    dict[tuple[str, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    tuple[int, int],
]:
    """Return exact numerator/denominator digit bounds for derived rows."""

    cell_den: dict[tuple[str, int], int] = {}
    cell_num: dict[tuple[str, int], int] = {}
    cell_operands: dict[tuple[str, int], int] = {}
    edge_den: dict[tuple[int, int], int] = {}
    edge_num: dict[tuple[int, int], int] = {}
    edge_operands: dict[tuple[int, int], int] = {}
    edge_sole_amount: dict[tuple[int, int], CanonicalRational] = {}
    for entry in flow.entries:
        amount_num = len(entry.amount.num)
        amount_den = len(entry.amount.den)
        edge_key = (entry.source, entry.target)
        edge_den[edge_key] = edge_den.get(edge_key, 0) + amount_den
        edge_num[edge_key] = max(edge_num.get(edge_key, 0), amount_num)
        edge_operands[edge_key] = edge_operands.get(edge_key, 0) + 1
        edge_sole_amount.setdefault(edge_key, entry.amount)
        for vertex in (entry.source, entry.target):
            cell_key = (entry.commodity_id, vertex)
            cell_den[cell_key] = cell_den.get(cell_key, 0) + amount_den
            cell_num[cell_key] = max(cell_num.get(cell_key, 0), amount_num)
            cell_operands[cell_key] = cell_operands.get(cell_key, 0) + 1

    def _sum_bound(operand_count: int, den_total: int, max_num: int) -> tuple[int, int]:
        # One operand is its own reduced sum: the component equals that
        # already-canonical operand and charges no summation growth.
        if operand_count == 1:
            return max_num, den_total
        return den_total + max_num + _DERIVED_DIGIT_SLACK, den_total

    cell_bounds: dict[tuple[str, int], tuple[int, int]] = {}
    for commodity in flow.commodities:
        for vertex in range(flow.network.vertex_count):
            key = (commodity.commodity_id, vertex)
            if key in cell_den:
                cell_bounds[key] = _sum_bound(
                    cell_operands[key], cell_den[key], cell_num[key]
                )
            else:
                cell_bounds[key] = (1, 1)

    load_bounds: dict[tuple[int, int], tuple[int, int]] = {}
    slack_bounds: dict[tuple[int, int], tuple[int, int]] = {}
    congestion_bound = (1, 1)
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        capacity_num = len(edge.capacity.num)
        capacity_den = len(edge.capacity.den)
        den_total = edge_den.get(edge_key)
        if den_total is None:
            # No amount reaches this edge: its load is exactly the zero
            # rational, its slack is exactly its capacity, and its congestion
            # ratio reduces to exactly zero.
            load_bounds[edge_key] = (1, 1)
            slack_bounds[edge_key] = (capacity_num, capacity_den)
            continue
        load_n, load_d = _sum_bound(
            edge_operands[edge_key], den_total, edge_num[edge_key]
        )
        if edge_operands[edge_key] == 1 and edge_sole_amount[edge_key] == edge.capacity:
            # The lone amount equals the capacity exactly: the load is the
            # capacity, the slack is the exact zero rational, and the
            # congestion ratio reduces to exactly one.
            load_bounds[edge_key] = (load_n, load_d)
            slack_bounds[edge_key] = (1, 1)
            continue
        slack_n = max(capacity_num + den_total, load_n + capacity_den) + 1
        slack_d = capacity_den + den_total
        load_bounds[edge_key] = (load_n, load_d)
        slack_bounds[edge_key] = (slack_n, slack_d)
        # The ratio divides one load by one capacity: both sides pick up the
        # capacity numerator only.
        congestion_bound = (
            max(congestion_bound[0], load_n + capacity_num),
            max(congestion_bound[1], den_total + capacity_num),
        )
    return cell_bounds, load_bounds, slack_bounds, congestion_bound


def derived_profile_digit_budget(flow: MulticommodityFlow) -> int:
    """Return the exact digit bound shared by every derived profile component."""

    cell_bounds, load_bounds, slack_bounds, congestion_bound = (
        _profile_component_digit_bounds(flow)
    )
    component_sides = [
        *cell_bounds.values(),
        *load_bounds.values(),
        *slack_bounds.values(),
        congestion_bound,
    ]
    return max(max(sides) for sides in component_sides)


def _require_derived_digit_budget(flow: MulticommodityFlow) -> int:
    """Reject operands whose implied profile cannot remain a canonical value."""

    budget = derived_profile_digit_budget(flow)
    if budget > MAX_CANONICAL_RATIONAL_DIGITS:
        raise ValueError(
            "multicommodity-flow arithmetic operands imply a "
            f"{budget}-digit derived-profile bound above the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit canonical cap"
        )
    return budget


def _require_profile_output_admission(flow: MulticommodityFlow) -> None:
    """Reject tensors whose echoed source and priced rows exceed the envelope."""

    cell_bounds, load_bounds, slack_bounds, congestion_bound = (
        _profile_component_digit_bounds(flow)
    )
    source_bytes = len(encode_strict_json(flow.model_dump(mode="json")))
    divergence_bytes = sum(
        num_digits
        + den_digits
        + _RATIONAL_JSON_OVERHEAD_BYTES
        + _DIVERGENCE_ROW_OVERHEAD_BYTES
        for num_digits, den_digits in cell_bounds.values()
    )
    edge_bytes = 0
    congestion_bytes = sum(congestion_bound) + _RATIONAL_JSON_OVERHEAD_BYTES
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
        source_bytes
        + divergence_bytes
        + edge_bytes
        + congestion_bytes
        + _PROFILE_RESULT_HEADER_BYTES
    )
    if estimated_bytes > MAX_PROFILE_RESULT_BYTES:
        raise ValueError(
            "multicommodity-flow profile result would exceed the "
            f"{MAX_PROFILE_RESULT_BYTES}-byte aggregate result bound"
        )


class MulticommodityFlow(StrictModel):
    """A canonical sparse exact commodity-by-edge tensor over one FlowGraph.

    Every omitted tensor cell denotes exact zero.  Graph edges, commodities, and
    nonzero tensor entries are sorted so the value has one JSON representation.
    The source network and demand tuple remain attached to the tensor, allowing
    downstream operations to consume this value without reconstructing context.
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
        _require_derived_digit_budget(self)
        _require_profile_output_admission(self)
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
            raise ValueError("result must match the exact multicommodity-flow profile")
        return self


__all__ = [
    "MAX_COMMODITY_VERTEX_CELLS",
    "MAX_MULTICOMMODITY_EDGES",
    "MAX_PROFILE_ADDITIONS_PER_PASS",
    "MAX_PROFILE_COMPARISONS_PER_PASS",
    "MAX_PROFILE_DIVISIONS_PER_PASS",
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
