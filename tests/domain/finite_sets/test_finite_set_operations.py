from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.finite_sets import finite_set_operations


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", finite_set_operations()) as services:
        yield services


def test_symmetric_difference_is_canonicalized(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="finite_set.compute.symmetric_difference",
            input={
                "left": {"elements": ["3", "1"]},
                "right": {"elements": ["2", "3"]},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"elements": ["1", "2"]}
