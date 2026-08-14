from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import combinatorics_operations


@pytest.fixture
def combinatorics_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state",
        combinatorics_operations(),
    ) as services:
        yield services


def test_source_scale_binomial_completes_independent_verification(
    combinatorics_services: DomainTestServices,
) -> None:
    payload = {"n": 1912, "k": 16}
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.compute.binomial",
            input=payload,
        )
    )
    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.binomial.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.output["result"]["value"] == (
        "1431712059377249479518540967853195958045"
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "combinatorics.compute.binomial"
    assert verified.verification_record_uri is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_public_binomial_checker_rejects_a_forged_value(
    combinatorics_services: DomainTestServices,
) -> None:
    rejected = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.binomial.verify",
            input={
                "input": {"n": 25, "k": 12},
                "candidate": {"value": "5200301"},
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
    assert rejected.verification_record_uri is None


def test_binomial_checker_accepts_zero_when_k_exceeds_n(
    combinatorics_services: DomainTestServices,
) -> None:
    payload = {"n": 5, "k": 7}
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.compute.binomial",
            input=payload,
        )
    )
    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.binomial.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )

    assert computed.output["result"]["value"] == "0"
    assert verified.output["status"] == "VERIFIED"
