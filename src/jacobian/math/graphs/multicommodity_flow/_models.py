"""Canonical values and source-bound profiles for exact multicommodity flow."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictStr, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json
from jacobian.math.graphs.flow._models import FlowGraph

# This operation scans a sparse commodity-by-edge tensor and materializes one
# divergence value for every commodity/vertex pair.  The bounds are deliberately
# independent: a dense tensor has K*E cells, while the returned divergence table
# has K*V cells.  The sparse payload itself is capped separately.
MAX_MULTICOMMODITY_VERTICES = 32
MAX_MULTICOMMODITY_EDGES = 128
MAX_MULTICOMMODITIES = 16
MAX_COMMODITY_EDGE_CELLS = 2_048
MAX_COMMODITY_VERTEX_CELLS = 512
MAX_SPARSE_FLOW_ENTRIES = 128
MAX_FLOW_RATIONAL_DIGITS = 32

# A load or divergence can sum at most 128 reduced 32-digit rationals. Before
# reduction, one common denominator has at most 128*32 digits; the sum and one
# capacity subtraction add at most three decimal digits, and a congestion ratio
# multiplies one further 32-digit capacity component. 4,132 is therefore a
# conservative strict bound below CanonicalRational's global 32,768-digit cap.
MAX_PROFILE_RATIONAL_DIGITS = 4_132

# A result echoes its source tensor, then includes at most 512 divergence rows
# and 128 edge rows. A derived rational can occupy at most 2*d+24 canonical
# JSON bytes for d decimal digits; the conservative row overhead reserves ASCII
# keys, labels, separators, and vertices. At the declared maxima this estimate
# is below 6.6 MiB, leaving explicit headroom under the 10 MiB transport limit.
MAX_PROFILE_RESULT_BYTES = 8 * 1024 * 1024
_DERIVED_RATIONAL_WIRE_BYTES = 2 * MAX_PROFILE_RATIONAL_DIGITS + 24
_DIVERGENCE_ROW_OVERHEAD_BYTES = 128
_EDGE_ROW_OVERHEAD_BYTES = 128
_PROFILE_RESULT_HEADER_BYTES = 1_024

# One pass performs at most 3F+E additions/subtractions, K negations, E
# divisions, and K*V+3E comparisons. With the admitted maxima this is 1,552
# logical steps; producer plus exact result replay therefore costs at most 3,104.
MAX_PROFILE_LOGICAL_STEPS = 3_104
MAX_PROFILE_ADDITIONS_PER_PASS = 512
MAX_PROFILE_NEGATIONS_PER_PASS = 16
MAX_PROFILE_DIVISIONS_PER_PASS = 128
MAX_PROFILE_COMPARISONS_PER_PASS = 896


def _require_flow_rational(value: CanonicalRational, label: str) -> None:
    require_bounded_rational(
        value,
        max_digits=MAX_FLOW_RATIONAL_DIGITS,
        label=label,
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
    source: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    sink: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    demand: CanonicalRational

    @model_validator(mode="after")
    def require_distinct_positive_terminals(self) -> Self:
        if self.source == self.sink:
            raise ValueError("commodity source and sink must be distinct")
        if self.demand.as_fraction() <= 0:
            raise ValueError("commodity demand must be strictly positive")
        _require_flow_rational(self.demand, "commodity demand")
        return self


class CommodityEdgeFlow(StrictModel):
    """One positive sparse entry of a commodity-by-directed-edge flow tensor."""

    commodity_id: StrictStr = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
    )
    source: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    target: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    amount: CanonicalRational

    @model_validator(mode="after")
    def require_positive_bounded_amount(self) -> Self:
        if self.amount.as_fraction() <= 0:
            raise ValueError("sparse flow entries must have strictly positive amounts")
        _require_flow_rational(self.amount, "flow amount")
        return self


def _require_canonical_network(network: FlowGraph) -> tuple[tuple[int, int], ...]:
    if network.vertex_count > MAX_MULTICOMMODITY_VERTICES:
        raise ValueError(
            "multicommodity flow networks may have at most "
            f"{MAX_MULTICOMMODITY_VERTICES} vertices"
        )
    if len(network.edges) > MAX_MULTICOMMODITY_EDGES:
        raise ValueError(
            "multicommodity flow networks may have at most "
            f"{MAX_MULTICOMMODITY_EDGES} edges"
        )
    edge_keys = tuple((edge.source, edge.target) for edge in network.edges)
    if edge_keys != tuple(sorted(edge_keys)):
        raise ValueError("network edges must be sorted by (source, target)")
    for edge in network.edges:
        _require_flow_rational(edge.capacity, "network capacity")
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


def _require_profile_output_admission(flow: MulticommodityFlow) -> None:
    source_bytes = len(encode_strict_json(flow.model_dump(mode="json")))
    divergence_bytes = (
        len(flow.commodities)
        * flow.network.vertex_count
        * (_DERIVED_RATIONAL_WIRE_BYTES + _DIVERGENCE_ROW_OVERHEAD_BYTES)
    )
    edge_bytes = len(flow.network.edges) * (
        2 * _DERIVED_RATIONAL_WIRE_BYTES + _EDGE_ROW_OVERHEAD_BYTES
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
        max_length=MAX_MULTICOMMODITIES,
        description=(
            "Distinct commodity records sorted lexicographically by commodity_id."
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
        _require_profile_output_admission(self)
        return self


class MulticommodityFlowProfileRequest(StrictModel):
    """Compute one exact conservation, load, slack, and congestion profile."""

    flow: MulticommodityFlow = Field(
        description=(
            "Canonical sparse tensor with at most 16 commodities, 128 nonzero "
            "entries, 2,048 conceptual commodity-edge cells, 512 returned "
            "commodity-vertex cells, 32-digit input rationals, and an admitted "
            "aggregate result envelope below 8 MiB."
        )
    )


class CommodityDivergence(StrictModel):
    """The exact outgoing-minus-incoming flow for one commodity at one vertex."""

    commodity_id: StrictStr
    vertex: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    divergence: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_divergence(self) -> Self:
        require_bounded_rational(
            self.divergence,
            max_digits=MAX_PROFILE_RATIONAL_DIGITS,
            label="derived multicommodity-flow divergence",
        )
        return self


class EdgeLoadProfile(StrictModel):
    """Exact aggregate load and signed capacity slack for one directed edge."""

    source: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    target: int = Field(ge=0, le=MAX_MULTICOMMODITY_VERTICES - 1)
    load: CanonicalRational
    slack: CanonicalRational

    @model_validator(mode="after")
    def require_bounded_profile_values(self) -> Self:
        for value, label in ((self.load, "load"), (self.slack, "slack")):
            require_bounded_rational(
                value,
                max_digits=MAX_PROFILE_RATIONAL_DIGITS,
                label=f"derived multicommodity-flow {label}",
            )
        return self


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

        if self.congestion is not None:
            require_bounded_rational(
                self.congestion,
                max_digits=MAX_PROFILE_RATIONAL_DIGITS,
                label="derived multicommodity-flow congestion",
            )
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
    "MAX_FLOW_RATIONAL_DIGITS",
    "MAX_MULTICOMMODITIES",
    "MAX_MULTICOMMODITY_EDGES",
    "MAX_MULTICOMMODITY_VERTICES",
    "MAX_PROFILE_ADDITIONS_PER_PASS",
    "MAX_PROFILE_COMPARISONS_PER_PASS",
    "MAX_PROFILE_DIVISIONS_PER_PASS",
    "MAX_PROFILE_LOGICAL_STEPS",
    "MAX_PROFILE_NEGATIONS_PER_PASS",
    "MAX_PROFILE_RATIONAL_DIGITS",
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
]
