from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.rationals import rational_payload as _q
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization import graph_optimization_operations


@pytest.fixture
def graph_optimization_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    """Install graph optimization and its exact checkers without a portfolio."""

    with open_exact_domain_services(
        tmp_path / "state",
        graph_optimization_operations(),
    ) as services:
        yield services


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
    graph_optimization_services,
) -> None:
    payload = _connected_payload()
    computed = graph_optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.spanning_tree.minimum.compute",
            input=payload,
        )
    )
    verified = graph_optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.spanning_tree.minimum.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.output["result"]["total_weight"] == _q(3)
    assert computed.artifact_uris == ()
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["conclusion"] == "TRUE"
    assert verified.output["operation_id"] == ("graph.spanning_tree.minimum.compute")
    assert verified.output["verification_record_uri"] is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris
    assert verified.execution.detail == (
        "independent fundamental-cycle optimality certificate replay accepted "
        "graph.spanning_tree.minimum.compute"
    )


def test_minimum_spanning_tree_verifier_rejects_a_feasible_nonminimum_tree(
    graph_optimization_services,
) -> None:
    payload = _connected_payload()
    computed = graph_optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.spanning_tree.minimum.compute",
            input=payload,
        )
    )
    forged_candidate = deepcopy(computed.output["result"])
    forged_candidate["tree_edges"] = [
        _edge("a", "b", 1),
        _edge("a", "d", 4),
        _edge("b", "c", 1),
    ]
    forged_candidate["total_weight"] = _q(6)
    forged_candidate["optimality_certificate"]["checks"][0]["non_tree_edge"] = [
        "a",
        "c",
    ]
    forged_candidate["optimality_certificate"]["checks"][0]["edge_weight"] = _q(2)
    forged_candidate["optimality_certificate"]["checks"][0]["tree_path_vertices"] = [
        "a",
        "b",
        "c",
    ]
    forged_candidate["optimality_certificate"]["checks"][0][
        "maximum_tree_path_weight"
    ] = _q(1)
    forged_candidate["optimality_certificate"]["checks"][1]["non_tree_edge"] = [
        "c",
        "d",
    ]
    forged_candidate["optimality_certificate"]["checks"][1]["edge_weight"] = _q(1)
    forged_candidate["optimality_certificate"]["checks"][1]["tree_path_vertices"] = [
        "c",
        "b",
        "a",
        "d",
    ]
    forged_candidate["optimality_certificate"]["checks"][1][
        "maximum_tree_path_weight"
    ] = _q(4)

    rejected = graph_optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.spanning_tree.minimum.verify",
            input={"input": payload, "candidate": forged_candidate},
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None


def test_disconnected_no_spanning_tree_result_is_completely_replayed(
    graph_optimization_services,
) -> None:
    payload = {
        "graph": {
            "vertices": ["a", "b", "c"],
            "edges": [_edge("a", "b", -1)],
        }
    }
    computed = graph_optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.spanning_tree.minimum.compute",
            input=payload,
        )
    )
    verified = graph_optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="graph.spanning_tree.minimum.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.output["result"]["status"] == "NO_SPANNING_TREE"
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None
    assert verified.execution.detail == (
        "independent finite connectivity replay accepted "
        "graph.spanning_tree.minimum.compute"
    )
