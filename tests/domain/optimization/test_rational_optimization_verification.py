from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.optimization import rational_optimization_operations


def _q(num: int, den: int = 1) -> dict[str, str]:
    return {"num": str(num), "den": str(den)}


def _program() -> dict[str, object]:
    return {
        "variables": ["x", "y"],
        "objective": [_q(1), _q(2)],
        "coefficients": [[_q(1), _q(1)]],
        "rhs": [_q(1)],
    }


@pytest.fixture
def optimization_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", rational_optimization_operations()
    ) as services:
        yield services


def test_rational_lp_result_uses_independent_exact_replay(
    optimization_services: DomainTestServices,
) -> None:
    payload = {"program": _program(), "wall_seconds": 10}
    computed = optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="optimization.linear.rational_optimum.compute", input=payload
        )
    )
    verified = optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="optimization.linear.rational_optimum.verify",
            input={"input": payload, "candidate": computed.output["result"]},
        )
    )
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"
    assert (
        verified.output["operation_id"]
        == "optimization.linear.rational_optimum.compute"
    )
    assert verified.verification_record_uri is not None
    assert verified.output["verification_record_uri"] in verified.artifact_uris


def test_rational_lp_verifier_rejects_objective_inequality(
    optimization_services: DomainTestServices,
) -> None:
    payload = {"program": _program(), "wall_seconds": 10}
    computed = optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="optimization.linear.rational_optimum.compute", input=payload
        )
    )
    candidate = deepcopy(computed.output["result"])
    candidate["dual_candidate"] = [_q(0)]
    candidate["dual_objective"] = _q(0)
    candidate["dual_slacks"] = [_q(1), _q(2)]
    rejected = optimization_services.core.operations.invoke(
        OperationRequest(
            operation_id="optimization.linear.rational_optimum.verify",
            input={"input": payload, "candidate": candidate},
        )
    )
    assert rejected.execution.status is ExecutionStatus.COMPLETED
    assert rejected.output["status"] == "REJECTED"
    assert rejected.output["conclusion"] == "UNKNOWN"
    assert rejected.verification_record_uri is None
