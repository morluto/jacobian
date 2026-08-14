from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
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
        tmp_path / "state",
        number_theory_operations(),
    ) as services:
        yield services


_TRANSVERSAL = {
    "moduli": [2, 4],
    "left": [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [3, -1],
    ],
    "right": [[0, 0]],
}


def test_finite_abelian_factorization_normalizes_and_verifies_transversal(
    number_theory_services: DomainTestServices,
) -> None:
    computed = number_theory_services.core.operations.invoke(
        OperationRequest(
            operation_id="finite_abelian_group.exact_factorization.compute",
            input=_TRANSVERSAL,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    assert result["moduli"] == [2, 4]
    assert result["normalized_left"] == [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
    ]
    assert result["normalized_right"] == [[0, 0]]
    assert result["group_order"] == 8
    assert result["pair_count"] == 8
    assert result["distinct_sum_count"] == 8
    assert result["representation_histogram"] == [
        {"representation_count": 1, "element_count": 8}
    ]
    assert result["is_exact_factorization"] is True
    assert result["first_missing"] is None
    assert result["first_duplicate"] is None
    assert computed.artifact_uris == ()

    verified = number_theory_services.core.operations.invoke(
        OperationRequest(
            operation_id="finite_abelian_group.exact_factorization.verify",
            input={"input": _TRANSVERSAL, "candidate": result},
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.verification_record_uri is not None


def test_finite_abelian_factorization_reports_and_checks_first_duplicate(
    number_theory_services: DomainTestServices,
) -> None:
    payload = {
        "moduli": [2],
        "left": [[0], [1]],
        "right": [[0], [1]],
    }
    computed = number_theory_services.core.operations.invoke(
        OperationRequest(
            operation_id="finite_abelian_group.exact_factorization.compute",
            input=payload,
        )
    )

    assert computed.execution.status is ExecutionStatus.COMPLETED
    result = computed.output["result"]
    assert result["representation_histogram"] == [
        {"representation_count": 2, "element_count": 2}
    ]
    assert result["is_exact_factorization"] is False
    assert result["first_missing"] is None
    assert result["first_duplicate"] == {
        "element": [0],
        "left": [0],
        "right": [0],
        "other_left": [1],
        "other_right": [1],
    }

    forged = deepcopy(result)
    forged["is_exact_factorization"] = True
    rejected = number_theory_services.core.operations.invoke(
        OperationRequest(
            operation_id="finite_abelian_group.exact_factorization.verify",
            input={"input": payload, "candidate": forged},
        )
    )

    assert rejected.execution.status is ExecutionStatus.ERROR
    assert rejected.diagnostics[0].code == "INVALID_EXACT_DOMAIN_INPUT"
    assert rejected.verification_record_uri is None
