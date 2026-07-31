from __future__ import annotations

from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus


def _edge(
    left: str,
    right: str,
    weight: int,
) -> dict[str, object]:
    return {
        "endpoints": [left, right],
        "weight": _q(weight),
    }


def _connected_payload() -> dict[str, object]:
    return {
        "graph": {
            "vertices": ["a", "b", "c", "d"],
            "edges": [
                _edge("a", "b", 1),
                _edge("b", "c", 1),
                _edge("c", "d", 1),
                _edge("a", "d", 4),
                _edge("a", "c", 2),
            ],
        }
    }


def test_weighted_minimum_spanning_tree_is_independently_verified(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.compute",
            input=_connected_payload(),
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert computed.output["result"]["total_weight"] == _q(3)
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["operation_id"] == ("graph.spanning_tree.minimum.compute")
    assert verified.output["verification_record_uri"] is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.execution.detail == (
        "independent fundamental-cycle optimality certificate replay accepted "
        "graph.spanning_tree.minimum.compute"
    )


def test_minimum_spanning_tree_verifier_rejects_a_feasible_nonminimum_tree(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.compute",
            input=_connected_payload(),
        )
    )
    installed = authorized_complete_runtime.portfolio.domain_bundles[
        "graph_optimization"
    ]
    false_result = authorized_complete_runtime.core.artifacts.put(
        schema_uri=installed.result_schema_uris["graph.spanning_tree.minimum.compute"],
        semantics_uri=installed.semantics_uri,
        parents=(computed.output["input_uri"],),
        payload={
            "result_schema_version": "1",
            "status": "EXACT",
            "vertices": ["a", "b", "c", "d"],
            "order": 4,
            "connected": True,
            "component_count": 1,
            "components": [["a", "b", "c", "d"]],
            "tree_edges": [
                _edge("a", "b", 1),
                _edge("a", "d", 4),
                _edge("b", "c", 1),
            ],
            "total_weight": _q(6),
            "optimality_certificate": {
                "certificate_schema_version": "1",
                "method": "ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING",
                "checks": [
                    {
                        "non_tree_edge": ["a", "c"],
                        "edge_weight": _q(2),
                        "tree_path_vertices": ["a", "b", "c"],
                        "maximum_tree_path_weight": _q(1),
                        "condition": "EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT",
                    },
                    {
                        "non_tree_edge": ["c", "d"],
                        "edge_weight": _q(1),
                        "tree_path_vertices": ["c", "b", "a", "d"],
                        "maximum_tree_path_weight": _q(4),
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
        },
        summary="adversarial feasible but nonminimum spanning tree",
    )

    rejected = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": false_result.artifact_uri},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_disconnected_no_spanning_tree_result_is_completely_replayed(
    authorized_complete_runtime,
) -> None:
    computed = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.compute",
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [_edge("a", "b", -1)],
                }
            },
        )
    )
    verified = authorized_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.spanning_tree.minimum.verify",
            mode=CapabilityMode.VERIFY,
            input={"result_uri": computed.output["result_uri"]},
        )
    )

    assert computed.output["result"]["status"] == "NO_SPANNING_TREE"
    assert verified.output["status"] == "VERIFIED"
    assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
    assert verified.execution.detail == (
        "independent finite connectivity replay accepted "
        "graph.spanning_tree.minimum.compute"
    )
