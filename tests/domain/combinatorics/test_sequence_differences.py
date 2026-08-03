from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.sequences import build_sequence_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", build_sequence_bundle()) as services:
        yield services


@pytest.mark.parametrize(
    ("capability_id", "values"),
    (
        ("sequence.compute.first_differences", ["7"]),
        ("sequence.compute.second_differences", ["7"]),
        ("sequence.compute.second_differences", ["7", "11"]),
    ),
)
def test_finite_differences_return_natural_empty_result(
    domain_services: DomainTestServices,
    capability_id: str,
    values: list[str],
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input={"values": values},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"values": []}
    assert result.artifact_uris == ()
