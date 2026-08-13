from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization.bundle import build_graph_optimization_bundle


@pytest.fixture
def graph_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", build_graph_optimization_bundle()
    ) as services:
        yield services


@pytest.mark.parametrize(
    ("graph", "decision"),
    [
        (
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            },
            "EXISTS",
        ),
        (
            {
                "vertices": ["c", "a", "b", "d"],
                "edges": [["c", "a"], ["c", "b"], ["c", "d"]],
            },
            "DOES_NOT_EXIST",
        ),
    ],
)
def test_hamiltonian_path_decision_has_independent_replay(
    graph_services: DomainTestServices,
    graph: dict[str, object],
    decision: str,
) -> None:
    computed = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.decide",
            input={"graph": graph},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.output["result"]["decision"] == decision
    assert "verification_capability_id" not in computed.output["result"]
    assert "verification_input_field" not in computed.output["result"]
    assert computed.artifact_uris == ()

    verified = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.verify",
            input={"input": {"graph": graph}, "candidate": computed.output["result"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_hamiltonian_checker_rejects_a_forged_negative_decision(
    graph_services: DomainTestServices,
) -> None:
    input_payload = {
        "graph": {
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
        }
    }
    computed = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.decide",
            input=input_payload,
        )
    )
    forged = deepcopy(computed.output["result"])
    forged["decision"] = "DOES_NOT_EXIST"
    forged["path"] = []

    checked = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.hamiltonian_path.verify",
            input={"input": input_payload, "candidate": forged},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
    assert checked.output["conclusion"] == "UNKNOWN"
