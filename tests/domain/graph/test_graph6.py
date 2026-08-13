from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization.invariant_bundle import (
    build_graph_invariant_bundle,
)


@pytest.fixture
def graph_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", build_graph_invariant_bundle()
    ) as services:
        yield services


def test_graph6_h24_decode_is_exact_and_independently_replayed(
    graph_services: DomainTestServices,
) -> None:
    payload = {"graph6": "W{CGW_@?Y??@?@?@_@??@??K_????G??C??B??@????_??B"}
    computed = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.encoding.graph6.decode.compute", input=payload
        )
    )
    decoded = computed.output["result"]
    assert decoded["order"] == 24
    assert len(decoded["edges"]) == 30
    assert max(decoded["degrees"]) == 3
    verified = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.encoding.graph6.decode.verify",
            input={"input": payload, "candidate": decoded},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_graph6_checker_rejects_wrong_edge_result(
    graph_services: DomainTestServices,
) -> None:
    payload = {"graph6": "Bw"}
    computed = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.encoding.graph6.decode.compute", input=payload
        )
    )
    forged = deepcopy(computed.output["result"])
    forged["edges"] = forged["edges"][:-1]
    forged["degrees"] = [2, 1, 1]
    rejected = graph_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="graph.encoding.graph6.decode.verify",
            input={"input": payload, "candidate": forged},
        )
    )
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.verification_record_uri is None
