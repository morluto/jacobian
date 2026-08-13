from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.matrix_lattice import build_matrix_bundle


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _matrix(entries: list[list[int]]) -> dict[str, object]:
    return {"entries": [[_q(value) for value in row] for row in entries]}


def _truncated_legendre_matrix(prime: int) -> dict[str, object]:
    order = (prime - 5) // 2

    def entry(row: int, column: int) -> int:
        residue = (row - column) % prime
        if residue == 0:
            character = 0
        else:
            character = 1 if pow(residue, (prime - 1) // 2, prime) == 1 else -1
        return 1 + character

    return _matrix(
        [[entry(row, column) for column in range(order)] for row in range(order)]
    )


@pytest.fixture
def matrix_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", build_matrix_bundle()
    ) as services:
        yield services


def test_order_33_determinant_computes_and_verifies(
    matrix_services: DomainTestServices,
) -> None:
    payload = {"matrix": _truncated_legendre_matrix(71)}
    computed = matrix_services.core.capabilities.invoke(
        CapabilityRequest(capability_id="matrix.determinant.compute", input=payload)
    )
    verified = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.output["result"]["determinant"] == _q(529)
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_determinant_rejects_order_above_64(
    matrix_services: DomainTestServices,
) -> None:
    matrix = _matrix(
        [[1 if row == column else 0 for column in range(65)] for row in range(65)]
    )
    result = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.determinant.compute",
            input={"matrix": matrix},
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
