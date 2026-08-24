"""Canonical values and source-bound profiles for exact multicommodity flow."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictStr, model_validator

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.graphs.flow._models import FlowGraph

# This operation scans a sparse commodity-by-edge tensor and materializes one
# divergence value for every commodity/vertex pair.  Vertex count is owned by
# FlowGraph; admission controls the dense divergence table through the
# commodity-vertex cell budget, the sparse tensor through the entry and
# commodity-edge cell budgets, and the whole returned value through the
# aggregate result envelope below.  The commodity count is bounded by those
# same derived quantities rather than an independent fixed ceiling.
MAX_MULTICOMMODITY_EDGES = 128
MAX_COMMODITY_EDGE_CELLS = 2_048
MAX_COMMODITY_VERTEX_CELLS = 512
MAX_SPARSE_FLOW_ENTRIES = 128

# Entry amounts and edge capacities are the only arithmetic operands, and each
# derived component is computed independently from only the operands that can
# reach it: a divergence cell sums the amounts incident to one
# commodity/vertex pair, an edge load sums the amounts carried by that edge,
# a slack subtracts one capacity from one such load, and the congestion ratio
# divides one load by one capacity.  Adding one operand to a running sum
# contributes at most its numerator and denominator digits plus three carry
# digits, and the slack subtraction and congestion division each compose one
# further operand, so eight slack digits on top of a component's own totaled
# operand digits bound that component's intermediates and derived numerator
# and denominator exactly.  Demands take part only in exact conservation
# comparisons, never in arithmetic, so they are covered by the measured
# source echo instead of these budgets.
_DERIVED_DIGIT_SLACK = 8

# A result echoes its source tensor, then includes at most 512 divergence rows
# and 128 edge rows. A derived rational occupies at most 2*d+24 canonical JSON
# bytes when its own component bound limits d decimal digits per component;
# the conservative row overhead reserves ASCII keys, labels, separators, and
# vertices. Admission measures the echoed source exactly and prices every
# divergence row and edge row from its own component's digit bound against
# this aggregate envelope, keeping the serialized result inside the envelope
# with headroom under the 10 MiB transport limit.
MAX_PROFILE_RESULT_BYTES = 8 * 1024 * 1024
_DIVERGENCE_ROW_OVERHEAD_BYTES = 128
_EDGE_ROW_OVERHEAD_BYTES = 128
_PROFILE_RESULT_HEADER_BYTES = 1_024

# One pass performs at most 3F+E additions/subtractions, K negations, E
# divisions, and K*V+3E comparisons. Every admitted commodity occupies V >= 2
# distinct commodity-vertex cells because its source and sink differ, so the
# 512-cell divergence budget admits at most 256 commodities, one negation
# each. With the admitted maxima this is 1,792 logical steps; producer plus
# exact result replay therefore costs at most 3,584.
MAX_PROFILE_LOGICAL_STEPS = 3_584
MAX_PROFILE_ADDITIONS_PER_PASS = 512
MAX_PROFILE_NEGATIONS_PER_PASS = MAX_COMMODITY_VERTEX_CELLS // 2
MAX_PROFILE_DIVISIONS_PER_PASS = 128
MAX_PROFILE_COMPARISONS_PER_PASS = 896


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
    if len(network.edges) > MAX_MULTICOMMODITY_EDGES:
        raise ValueError(
            "multicommodity flow networks may have at most "
            f"{MAX_MULTICOMMODITY_EDGES} edges"
        )
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
    if len(commodities) * len(network.edges) > MAX_COMMODITY_EDGE_CELLS:
        raise ValueError(
            f"commodity-by-edge cell count exceeds {MAX_COMMODITY_EDGE_CELLS}"
        )
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


def _operand_digits(value: CanonicalRational) -> int:
    return len(value.num) + len(value.den)


def _profile_component_digit_bounds(
    flow: MulticommodityFlow,
) -> tuple[
    dict[tuple[str, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    """Return exact divergence-cell, edge-load, and edge-slack digit bounds."""

    cell_bounds: dict[tuple[str, int], int] = {}
    load_bounds: dict[tuple[int, int], int] = {}
    slack_bounds: dict[tuple[int, int], int] = {}
    for edge in flow.network.edges:
        edge_key = (edge.source, edge.target)
        load_bounds[edge_key] = _DERIVED_DIGIT_SLACK
        slack_bounds[edge_key] = _DERIVED_DIGIT_SLACK + _operand_digits(edge.capacity)
    for entry in flow.entries:
        digits = _operand_digits(entry.amount)
        edge_key = (entry.source, entry.target)
        load_bounds[edge_key] += digits
        slack_bounds[edge_key] += digits
        for vertex in (entry.source, entry.target):
            key = (entry.commodity_id, vertex)
            cell_bounds[key] = cell_bounds.get(key, _DERIVED_DIGIT_SLACK) + digits
    for commodity in flow.commodities:
        for vertex in range(flow.network.vertex_count):
            cell_bounds.setdefault(
                (commodity.commodity_id, vertex),
                _DERIVED_DIGIT_SLACK,
            )
    return cell_bounds, load_bounds, slack_bounds


def derived_profile_digit_budget(flow: MulticommodityFlow) -> int:
    """Return the exact digit bound shared by every derived profile component."""

    cell_bounds, load_bounds, slack_bounds = _profile_component_digit_bounds(flow)
    return max(
        max(cell_bounds.values(), default=_DERIVED_DIGIT_SLACK),
        max(load_bounds.values(), default=_DERIVED_DIGIT_SLACK),
        max(slack_bounds.values(), default=_DERIVED_DIGIT_SLACK),
    )


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

    cell_bounds, load_bounds, slack_bounds = _profile_component_digit_bounds(flow)
    source_bytes = len(encode_strict_json(flow.model_dump(mode="json")))
    divergence_bytes = sum(
        2 * bound + 24 + _DIVERGENCE_ROW_OVERHEAD_BYTES
        for bound in cell_bounds.values()
    )
    edge_bytes = sum(
        2 * load_bounds[(edge.source, edge.target)]
        + 24
        + 2 * slack_bounds[(edge.source, edge.target)]
        + 24
        + _EDGE_ROW_OVERHEAD_BYTES
        for edge in flow.network.edges
    )
    estimated_bytes = (
        source_bytes + divergence_bytes + edge_bytes + _PROFILE_RESULT_HEADER_BYTES
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
            "2,048 conceptual commodity-edge cells, 512 returned "
            "commodity-vertex cells (hence at most 256 commodities), 128 "
            "nonzero entries, a per-component exact digit budget derived from "
            "each component's own operands, and an admitted aggregate result "
            "envelope below 8 MiB."
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
    "MAX_COMMODITY_EDGE_CELLS",
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
