from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import combinatorics_operations


@pytest.fixture
def combinatorics_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state", combinatorics_operations()
    ) as services:
        yield services


def test_integer_partition_enumeration_is_complete_and_canonical(
    combinatorics_services: DomainTestServices,
) -> None:
    result = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id="combinatorics.enumerate.integer_partitions",
            input={"n": 5, "max_parts": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "n": 5,
        "max_parts": 2,
        "partitions": [[5], [4, 1], [3, 2]],
    }
