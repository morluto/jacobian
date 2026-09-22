"""Exact path-and-cycle decomposition of multicommodity tensors."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import pairwise

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import encode_strict_json, parse_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.flows._models import CapacitatedEdge, FlowGraph
from jacobian.math.graphs.flows.multicommodity._models import (
    CommodityDemand,
    CommodityEdgeFlow,
    MulticommodityFlow,
    MulticommodityFlowDecompositionRequest,
    MulticommodityFlowDecompositionResult,
)
from jacobian.math.graphs.flows.multicommodity._tools import (
    TOOLS,
    _run_multicommodity_flow_decomposition,
)
from jacobian.math.graphs.flows.multicommodity.operations import (
    decompose_multicommodity_flow,
)


def q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def _reconstruct(
    result: MulticommodityFlowDecompositionResult,
) -> dict[tuple[str, int, int], Fraction]:
    entries: dict[tuple[str, int, int], Fraction] = defaultdict(Fraction)
    for path_term in result.paths:
        for source, target in pairwise(path_term.vertices):
            entries[(path_term.commodity_id, source, target)] += (
                path_term.amount.as_fraction()
            )
    for cycle_term in result.cycles:
        for source, target in pairwise(cycle_term.vertices):
            entries[(cycle_term.commodity_id, source, target)] += (
                cycle_term.amount.as_fraction()
            )
    return dict(entries)


def _tensor(flow: MulticommodityFlow) -> dict[tuple[str, int, int], Fraction]:
    return {
        (entry.commodity_id, entry.source, entry.target): entry.amount.as_fraction()
        for entry in flow.entries
    }


def _path_cycle_flow() -> MulticommodityFlow:
    return MulticommodityFlow(
        network=FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(2)),
                CapacitatedEdge(source=1, target=2, capacity=q(2)),
                CapacitatedEdge(source=1, target=3, capacity=q(2)),
                CapacitatedEdge(source=2, target=1, capacity=q(2)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),
            CommodityEdgeFlow(commodity_id="a", source=1, target=2, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=1, target=3, amount=q(1)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=1, amount=q(1, 2)),
        ),
    )


def test_path_and_cycle_terms_reconstruct_fractional_tensor() -> None:
    result = decompose_multicommodity_flow(_path_cycle_flow())

    assert [(term.vertices, term.amount.as_fraction()) for term in result.paths] == [
        ((0, 1, 3), Fraction(1))
    ]
    assert [(term.vertices, term.amount.as_fraction()) for term in result.cycles] == [
        ((1, 2, 1), Fraction(1, 2))
    ]
    restored = MulticommodityFlowDecompositionResult.model_validate_json(
        result.model_dump_json()
    )
    assert _reconstruct(restored) == _tensor(restored.flow)


def test_empty_support_round_trips_as_empty_decomposition() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=(),
    )

    result = decompose_multicommodity_flow(flow)
    restored = MulticommodityFlowDecompositionResult.model_validate_json(
        result.model_dump_json()
    )
    assert result.paths == ()
    assert result.cycles == ()
    assert _reconstruct(restored) == {}
    assert restored == result


def test_pure_circulation_is_retained_as_a_cycle() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(2)),
                CapacitatedEdge(source=1, target=0, capacity=q(2)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=2, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(3, 5)),
            CommodityEdgeFlow(commodity_id="a", source=1, target=0, amount=q(3, 5)),
        ),
    )

    result = decompose_multicommodity_flow(flow)
    assert result.paths == ()
    assert [(term.vertices, term.amount.as_fraction()) for term in result.cycles] == [
        ((0, 1, 0), Fraction(3, 5))
    ]
    assert _reconstruct(result) == _tensor(flow)


def test_diamond_fractional_paths_are_deterministic_and_exact() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=0, target=2, capacity=q(1)),
                CapacitatedEdge(source=1, target=3, capacity=q(1)),
                CapacitatedEdge(source=2, target=3, capacity=q(1)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=1, target=3, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1, 2)),
        ),
    )

    first = decompose_multicommodity_flow(flow)
    second = decompose_multicommodity_flow(flow)
    assert first == second
    assert [term.vertices for term in first.paths] == [(0, 1, 3), (0, 2, 3)]
    assert all(term.amount == q(1, 2) for term in first.paths)
    assert first.cycles == ()
    assert _reconstruct(first) == _tensor(flow)


def test_circulation_adversary_uses_bounded_public_search() -> None:
    # The old recursive path search explored every simple trail through the
    # complete circulation before taking 1 -> 63.  A breadth-first residual
    # search must complete without enumerating those trails.
    circulation_vertices = tuple(range(2, 15))
    sink = 63
    edge_keys = [(0, 1), (1, 2), (2, 1), (1, sink)]
    edge_keys.extend(
        (source, target)
        for source in circulation_vertices
        for target in circulation_vertices
        if source != target
    )
    edge_keys.sort()
    network = FlowGraph(
        vertex_count=64,
        edges=tuple(
            CapacitatedEdge(source=source, target=target, capacity=q(100))
            for source, target in edge_keys
        ),
    )
    flow = MulticommodityFlow(
        network=network,
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=sink, demand=q(1)),
        ),
        entries=tuple(
            CommodityEdgeFlow(
                commodity_id="a", source=source, target=target, amount=q(1)
            )
            for source, target in edge_keys
        ),
    )

    result = decompose_multicommodity_flow(flow)

    assert len(result.paths) == 1
    assert result.paths[0].vertices == (0, 1, sink)
    assert _reconstruct(result) == _tensor(flow)


def test_catalog_and_native_decomposition_paths_agree() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "network.multicommodity_flow.decomposition.compute"
    )
    example = tool.examples[0]
    request = MulticommodityFlowDecompositionRequest.model_validate_json(
        encode_strict_json(example.input), strict=True
    )
    assert _run_multicommodity_flow_decomposition(
        request
    ) == decompose_multicommodity_flow(request.flow)


def test_native_boundary_rejects_forged_unknown_commodity() -> None:
    flow = _path_cycle_flow()
    forged_entry = CommodityEdgeFlow(
        commodity_id="forged",
        source=0,
        target=1,
        amount=q(1),
    )
    forged = flow.model_copy(update={"entries": (*flow.entries, forged_entry)})

    with pytest.raises(OperationDomainValidationError) as exc_info:
        decompose_multicommodity_flow(forged)

    assert exc_info.value.errors()[0]["type"] == (
        "graph.multicommodity_decomposition_invalid_canonical_flow"
    )


def test_serialized_result_rejects_missing_path_term() -> None:
    result = decompose_multicommodity_flow(_path_cycle_flow())
    payload = result.model_dump(mode="json")
    payload["paths"] = []
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_native_boundary_rejects_noncanonical_flow() -> None:
    with pytest.raises(OperationDomainValidationError):
        decompose_multicommodity_flow(object())  # type: ignore[arg-type]


def test_native_boundary_rejects_forged_edge_axis() -> None:
    flow = _path_cycle_flow()
    forged_network = flow.network.model_copy(update={"edges": (object(),)})
    forged = flow.model_copy(update={"network": forged_network})

    with pytest.raises(OperationDomainValidationError) as exc_info:
        decompose_multicommodity_flow(forged)

    assert exc_info.value.errors()[0]["type"] == (
        "graph.multicommodity_decomposition_invalid_canonical_flow"
    )


def test_denominator_fold_is_rejected_before_oversized_fraction() -> None:
    # Three pairwise-coprime canonical denominators force a roughly 66,000
    # digit common denominator at the source divergence.  The operation's
    # bounded addition guard must refuse before Fraction constructs it.
    width = MAX_CANONICAL_RATIONAL_DIGITS * 2 // 3 + 1
    denominators = tuple(
        parse_canonical_integer("1" + "0" * width + str(offset)) for offset in (1, 3, 7)
    )
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=0, target=2, capacity=q(1)),
                CapacitatedEdge(source=0, target=3, capacity=q(1)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=tuple(
            CommodityEdgeFlow(
                commodity_id="a",
                source=0,
                target=target,
                amount=CanonicalRational(num=1, den=denominator),
            )
            for target, denominator in zip((1, 2, 3), denominators, strict=True)
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        decompose_multicommodity_flow(flow)

    assert exc_info.value.errors()[0]["type"] == (
        "graph.multicommodity_decomposition_rational_envelope"
    )
