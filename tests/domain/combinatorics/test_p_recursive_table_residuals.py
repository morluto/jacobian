from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import combinatorics_operations


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


@pytest.fixture
def combinatorics_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", combinatorics_operations()
    ) as services:
        yield services


def _payload() -> dict[str, object]:
    return {
        "coefficient_polynomials": [[_q(1)], [_q(0), _q(-1)]],
        "values": [_q(value) for value in (1, 1, 2, 6, 24, 120)],
        "coefficient_convention": (
            "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
        ),
        "polynomial_convention": "ASCENDING_POWERS_OF_N",
        "table_convention": "VALUES_A_0_THROUGH_A_N_IN_ORDER",
    }


def test_submitted_table_residuals_compute_and_verify(
    combinatorics_services: DomainTestServices,
) -> None:
    payload = _payload()
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=(
                "combinatorics.recurrence.p_recursive.table_residuals.compute"
            ),
            input=payload,
        )
    )
    output = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert output["satisfies_recurrence"] is True
    assert output["first_failure_index"] is None
    assert [item["index"] for item in output["residuals"]] == [1, 2, 3, 4, 5]
    assert {item["value"]["num"] for item in output["residuals"]} == {"0"}

    verified = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=(
                "combinatorics.recurrence.p_recursive.table_residuals.verify"
            ),
            input={"input": payload, "candidate": output},
        )
    )
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_submitted_table_reports_failure_and_rejects_forged_success(
    combinatorics_services: DomainTestServices,
) -> None:
    payload = _payload()
    payload["values"][-1] = _q(121)
    computed = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=(
                "combinatorics.recurrence.p_recursive.table_residuals.compute"
            ),
            input=payload,
        )
    )
    output = computed.output["result"]
    assert output["satisfies_recurrence"] is False
    assert output["first_failure_index"] == 5

    forged = deepcopy(output)
    forged["residuals"][-1]["value"] = _q(0)
    forged["satisfies_recurrence"] = True
    forged["first_failure_index"] = None
    rejected = combinatorics_services.core.operations.invoke(
        OperationRequest(
            operation_id=(
                "combinatorics.recurrence.p_recursive.table_residuals.verify"
            ),
            input={"input": payload, "candidate": forged},
        )
    )
    assert rejected.output["status"] == "REJECTED"
    assert rejected.verification_record_uri is None
