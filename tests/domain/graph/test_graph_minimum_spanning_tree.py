from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError
from pytest import fixture, raises
from tests.support.rationals import rational_payload as _q
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import (
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.graph_optimization import (
    GraphMinimumSpanningTreeRequest,
    GraphMinimumSpanningTreeResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization.bundle import (
    build_graph_optimization_bundle,
)


@fixture
def graph_optimization_services(
    tmp_path: Path,
) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path,
        build_graph_optimization_bundle(),
    ) as services:
        yield services


def _edge(
    left: str,
    right: str,
    numerator: int | str,
    denominator: int | str = 1,
) -> dict[str, object]:
    return {
        "endpoints": [left, right],
        "weight": _q(numerator, denominator),
    }


def _weighted_graph(
    *,
    vertices: list[str],
    edges: list[dict[str, object]],
) -> dict[str, object]:
    return {"vertices": vertices, "edges": edges}


def _result_payload(_services: DomainTestServices, result: object) -> dict[str, object]:
    return result.output["result"]  # type: ignore[attr-defined, no-any-return]


def test_exact_weighted_minimum_spanning_tree_and_lineage(
    graph_optimization_services: DomainTestServices,
) -> None:
    result = graph_optimization_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.compute",
            input={
                "graph": _weighted_graph(
                    vertices=["d", "c", "b", "a"],
                    edges=[
                        _edge("d", "a", 5),
                        _edge("c", "a", 7, 2),
                        _edge("d", "c", 3),
                        _edge("c", "b", 2),
                        _edge("b", "a", 1),
                    ],
                )
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert _result_payload(graph_optimization_services, result) == {
        "result_schema_version": "1",
        "status": "EXACT",
        "vertices": ["a", "b", "c", "d"],
        "order": 4,
        "connected": True,
        "component_count": 1,
        "components": [["a", "b", "c", "d"]],
        "tree_edges": [
            _edge("a", "b", 1),
            _edge("b", "c", 2),
            _edge("c", "d", 3),
        ],
        "total_weight": _q(6),
        "optimality_certificate": {
            "certificate_schema_version": "1",
            "method": "ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING",
            "checks": [
                {
                    "non_tree_edge": ["a", "c"],
                    "edge_weight": _q(7, 2),
                    "tree_path_vertices": ["a", "b", "c"],
                    "maximum_tree_path_weight": _q(2),
                    "condition": "EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT",
                },
                {
                    "non_tree_edge": ["a", "d"],
                    "edge_weight": _q(5),
                    "tree_path_vertices": ["a", "b", "c", "d"],
                    "maximum_tree_path_weight": _q(3),
                    "condition": "EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT",
                },
            ],
            "required_checks": [
                "SOURCE_CONNECTIVITY",
                "TREE_SPANNING_ACYCLIC",
                "TOTAL_WEIGHT_EXACT",
                "ALL_NON_TREE_EDGES_COVERED",
                "CYCLE_NON_IMPROVEMENT",
            ],
        },
        "convention": (
            "MINIMUM_TOTAL_EDGE_WEIGHT_OVER_QQ_EMPTY_GRAPH_HAS_NO_SPANNING_TREE"
        ),
        "completion": "COMPLETE",
    }
    assert result.artifact_uris == ()


def test_disconnected_and_empty_graphs_have_complete_no_tree_outcomes(
    graph_optimization_services: DomainTestServices,
) -> None:
    for graph, expected_components in (
        (
            _weighted_graph(
                vertices=["a", "b", "c"],
                edges=[_edge("a", "b", -2)],
            ),
            [["a", "b"], ["c"]],
        ),
        (_weighted_graph(vertices=[], edges=[]), []),
    ):
        result = graph_optimization_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="graph.spanning_tree.minimum.compute",
                input={"graph": graph},
            )
        )

        assert result.execution.status is ExecutionStatus.COMPLETED
        computed = _result_payload(graph_optimization_services, result)
        assert computed["status"] == "NO_SPANNING_TREE"
        assert computed["components"] == expected_components
        assert computed["tree_edges"] == []
        assert computed["total_weight"] is None
        assert computed["completion"] == "COMPLETE"


def test_equal_weight_ties_are_deterministic_under_input_reordering(
    graph_optimization_services: DomainTestServices,
) -> None:
    first = _weighted_graph(
        vertices=["d", "c", "b", "a"],
        edges=[
            _edge("d", "a", 1),
            _edge("c", "d", 1),
            _edge("b", "c", 1),
            _edge("a", "b", 1),
        ],
    )
    second = _weighted_graph(
        vertices=["a", "b", "c", "d"],
        edges=[
            _edge("b", "a", 1),
            _edge("c", "b", 1),
            _edge("d", "c", 1),
            _edge("a", "d", 1),
        ],
    )

    outputs = [
        _result_payload(
            graph_optimization_services,
            graph_optimization_services.core.capabilities.invoke(
                CapabilityRequest(
                    capability_id="graph.spanning_tree.minimum.compute",
                    input={"graph": graph},
                )
            ),
        )
        for graph in (first, second)
    ]

    assert outputs[0] == outputs[1]
    assert outputs[0]["tree_edges"] == [
        _edge("a", "b", 1),
        _edge("a", "d", 1),
        _edge("b", "c", 1),
    ]


def test_weighted_mst_intent_is_discoverable_and_example_is_valid(
    graph_optimization_services: DomainTestServices,
) -> None:
    discovered = graph_optimization_services.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "compute an exact weighted minimum spanning tree with total "
                "weight and inspectable cycle optimality evidence"
            ),
            domain="graph",
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == (
        "graph.spanning_tree.minimum.compute"
    )
    assert discovered.matches[0].relevance_score > 0
    descriptor = next(
        descriptor
        for descriptor in graph_optimization_services.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "graph.spanning_tree.minimum.compute"
    )
    assert descriptor.invocation_examples[0].name == "four_vertex_weighted_graph"

    result = graph_optimization_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=descriptor.capability_id,
            input=descriptor.invocation_examples[0].input,
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert _result_payload(graph_optimization_services, result)["total_weight"] == _q(6)


def test_weighted_graph_contract_rejects_parallel_edges_and_oversized_weights() -> None:
    with raises(ValidationError, match="unique ignoring orientation"):
        GraphMinimumSpanningTreeRequest.model_validate(
            {
                "graph": _weighted_graph(
                    vertices=["a", "b"],
                    edges=[
                        _edge("a", "b", 1),
                        _edge("b", "a", 2),
                    ],
                )
            }
        )

    with raises(ValidationError, match="256-digit bound"):
        GraphMinimumSpanningTreeRequest.model_validate(
            {
                "graph": _weighted_graph(
                    vertices=["a", "b"],
                    edges=[_edge("a", "b", "1" * 257)],
                )
            }
        )


def test_invalid_weighted_graph_fails_before_artifact_writes(
    graph_optimization_services: DomainTestServices,
) -> None:
    result = graph_optimization_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.compute",
            input={
                "graph": _weighted_graph(
                    vertices=["a", "b"],
                    edges=[_edge("a", "missing", 1)],
                )
            },
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_MINIMUM_SPANNING_TREE_REQUEST"
    assert result.artifact_uris == ()


def test_minimum_spanning_tree_result_rejects_inconsistent_status() -> None:
    with raises(ValidationError, match="connected nonempty spanning tree"):
        GraphMinimumSpanningTreeResult.model_validate(
            {
                "status": "EXACT",
                "vertices": ["a", "b"],
                "order": 2,
                "connected": False,
                "component_count": 2,
                "components": [["a"], ["b"]],
                "tree_edges": [],
                "total_weight": None,
                "optimality_certificate": {"checks": []},
            }
        )
