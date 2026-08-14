from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.number_theory import number_theory_operations


@pytest.fixture
def number_theory_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", number_theory_operations()
    ) as services:
        yield services


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (("0", "0", "0"), ("0", "-72", "0"), ("-21", "6", "42"), ("60", "72", "360")),
)
def test_lcm_result_uses_independent_euclidean_replay(
    number_theory_services: DomainTestServices,
    left: str,
    right: str,
    expected: str,
) -> None:
    payload = {"left": left, "right": right}
    computed = number_theory_services.core.operations.invoke(
        OperationRequest(operation_id="integer.compute.lcm", input=payload)
    )
    verified = number_theory_services.core.operations.invoke(
        OperationRequest(
            operation_id="integer.lcm.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.output["result"]["value"] == expected
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "integer.compute.lcm"
    assert verified.verification_record_uri is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_lcm_verifier_rejects_schema_valid_wrong_value(
    number_theory_services: DomainTestServices,
) -> None:
    rejected = number_theory_services.core.operations.invoke(
        OperationRequest(
            operation_id="integer.lcm.verify",
            input={
                "input": {"left": "60", "right": "72"},
                "candidate": {"value": "720"},
            },
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.verification_record_uri is None
