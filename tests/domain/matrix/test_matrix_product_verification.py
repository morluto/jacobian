from __future__ import annotations

from copy import deepcopy

from tests.support.rationals import rational_payload as _q
from tests.support.services import DomainTestServices

from jacobian.contracts.capabilities import (
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.contracts.results import ExecutionStatus


def _qq(entries: list[list[int]]) -> dict[str, object]:
    return {
        "domain": "QQ",
        "entries": [[_q(value) for value in row] for row in entries],
    }


def _square_input() -> dict[str, object]:
    matrix = _qq([[0, 1], [0, 0]])
    return {"left": matrix, "right": matrix}


def _compute_square(matrix_services: DomainTestServices) -> CapabilityResult:
    return matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.multiply.compute",
            input=_square_input(),
        )
    )


def test_square_zero_product_is_computed_then_independently_verified(
    matrix_services: DomainTestServices,
) -> None:
    computed = _compute_square(matrix_services)

    assert computed.execution.status is ExecutionStatus.COMPLETED

    verified = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.multiply.verify",
            input={
                "input": _square_input(),
                "candidate": computed.output["result"],
            },
        )
    )

    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert verified.output["operation_id"] == "matrix.multiply.compute"
    assert verified.output["verification_record_uri"] is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_matrix_product_verifier_rejects_a_false_product_without_a_record(
    matrix_services: DomainTestServices,
) -> None:
    computed = _compute_square(matrix_services)
    false_candidate = deepcopy(computed.output["result"])
    false_candidate["product"] = _qq([[1, 0], [0, 0]])

    rejected = matrix_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.multiply.verify",
            input={
                "input": _square_input(),
                "candidate": false_candidate,
            },
        )
    )

    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.output["verification_record_uri"] is None
