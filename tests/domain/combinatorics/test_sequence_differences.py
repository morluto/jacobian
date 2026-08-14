from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.sequences import sequence_operations


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", sequence_operations()) as services:
        yield services


@pytest.mark.parametrize(
    ("operation_id", "values"),
    (
        ("sequence.compute.first_differences", ["7"]),
        ("sequence.compute.second_differences", ["7"]),
        ("sequence.compute.second_differences", ["7", "11"]),
    ),
)
def test_finite_differences_return_natural_empty_result(
    domain_services: DomainTestServices,
    operation_id: str,
    values: list[str],
) -> None:
    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id=operation_id,
            input={"values": values},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"values": []}
    assert result.artifact_uris == ()
